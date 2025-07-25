from __future__ import annotations

import datetime
import functools
import pytz
import io
import math
import os
from collections import namedtuple
import re

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, ImageColor, PngImagePlugin, ImageOps
from PIL import __version__ as pillow_version
from pkg_resources import parse_version
# pillow_avif needs to be imported somewhere in code for it to work
import pillow_avif # noqa: F401
import string
import json
import hashlib

from modules import sd_samplers, shared, script_callbacks, errors
from modules.paths_internal import roboto_ttf_file
from modules.shared import opts

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def get_font(fontsize: int):
    try:
        return ImageFont.truetype(opts.font or roboto_ttf_file, fontsize)
    except Exception:
        return ImageFont.truetype(roboto_ttf_file, fontsize)


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        if opts.n_rows > 0:
            rows = opts.n_rows
        elif opts.n_rows == 0:
            rows = batch_size
        elif opts.grid_prevent_empty_spots:
            rows = math.floor(math.sqrt(len(imgs)))
            while len(imgs) % rows != 0:
                rows -= 1
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)
    if rows > len(imgs):
        rows = len(imgs)

    cols = math.ceil(len(imgs) / rows)

    params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
    script_callbacks.image_grid_callback(params)

    w, h = map(max, zip(*(img.size for img in imgs)))
    grid_background_color = ImageColor.getcolor(opts.grid_background_color, 'RGBA')
    grid = Image.new('RGBA', size=(params.cols * w, params.rows * h), color=grid_background_color)

    for i, img in enumerate(params.imgs):
        img_w, img_h = img.size
        w_offset, h_offset = 0 if img_w == w else (w - img_w) // 2, 0 if img_h == h else (h - img_h) // 2
        grid.paste(img, box=(i % params.cols * w + w_offset, i // params.cols * h + h_offset))

    return grid


class Grid(namedtuple("_Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])):
    @property
    def tile_count(self) -> int:
        """
        The total number of tiles in the grid.
        """
        return sum(len(row[2]) for row in self.tiles)


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w, h = image.size

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None


def draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin=0):

    color_active = ImageColor.getcolor(opts.grid_text_active_color, 'RGB')
    color_inactive = ImageColor.getcolor(opts.grid_text_inactive_color, 'RGB')
    color_background = ImageColor.getcolor(opts.grid_background_color, 'RGB')

    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines, initial_fnt, initial_fontsize):
        for line in lines:
            fnt = initial_fnt
            fontsize = initial_fontsize
            if parse_version(pillow_version) >= parse_version('10.0.0'):
                # New code for Pillow 10.0.0+
                text_width, text_height = drawing.multiline_textbbox((0, 0), line.text, font=fnt)[2:]
                while text_width > line.allowed_width and fontsize > 0:
                    fontsize -= 1
                    fnt = get_font(fontsize)
                    text_width, text_height = drawing.multiline_textbbox((0, 0), line.text, font=fnt)[2:]
            else:
                # Old code for Pillow versions below 10.0.0
                while drawing.multiline_textsize(line.text, font=fnt)[0] > line.allowed_width and fontsize > 0:
                    fontsize -= 1
                    fnt = get_font(fontsize)

            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2

    fnt = get_font(fontsize)

    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else width * 3 // 4

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), color_background)
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            line.allowed_width = allowed_width

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]

    pad_top = 0 if sum(hor_text_heights) == 0 else max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left + margin * (cols-1), im.height + pad_top + margin * (rows-1)), color_background)

    for row in range(rows):
        for col in range(cols):
            cell = im.crop((width * col, height * row, width * (col+1), height * (row+1)))
            result.paste(cell, (pad_left + (width + margin) * col, pad_top + (height + margin) * row))

    d = ImageDraw.Draw(result)

    for col in range(cols):
        x = pad_left + (width + margin) * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col], fnt, fontsize)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + (height + margin) * row + height / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row], fnt, fontsize)

    return result


def draw_prompt_matrix(im, width, height, all_prompts, margin=0):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)

    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]

    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]

    return draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin)


def resize_image(resize_mode, im, width, height, upscaler_name=None, force_RGBA=False):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    upscaler_name = upscaler_name or opts.upscaler_for_img2img

    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L' or force_RGBA:
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            upscalers = [x for x in shared.sd_upscalers if x.name == upscaler_name]
            if len(upscalers) == 0:
                upscaler = shared.sd_upscalers[0]
                print(f"could not find upscaler named {upscaler_name or '<empty string>'}, using {upscaler.name} as a fallback")
            else:
                upscaler = upscalers[0]

            im = upscaler.scaler.upscale(im, scale, upscaler.data_path)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB" if not force_RGBA else "RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB" if not force_RGBA else "RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


if not shared.cmd_opts.unix_filenames_sanitization:
    invalid_filename_chars = '#<>:"/\\|?*\n\r\t'
else:
    invalid_filename_chars = '/'
invalid_filename_prefix = ' '
invalid_filename_postfix = ' .'
re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
max_filename_part_length = shared.cmd_opts.filenames_max_length
NOTHING_AND_SKIP_PREVIOUS_TEXT = object()


def sanitize_filename_part(text, replace_spaces=True):
    if text is None:
        return None

    if replace_spaces:
        text = text.replace(' ', '_')

    text = text.translate({ord(x): '_' for x in invalid_filename_chars})
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    text = text.rstrip(invalid_filename_postfix)
    return text


@functools.cache
def get_scheduler_str(sampler_name, scheduler_name):
    """Returns {Scheduler} if the scheduler is applicable to the sampler"""
    if scheduler_name == 'Automatic':
        config = sd_samplers.find_sampler_config(sampler_name)
        scheduler_name = config.options.get('scheduler', 'Automatic')
    return scheduler_name.capitalize()


@functools.cache
def get_sampler_scheduler_str(sampler_name, scheduler_name):
    """Returns the '{Sampler} {Scheduler}' if the scheduler is applicable to the sampler"""
    return f'{sampler_name} {get_scheduler_str(sampler_name, scheduler_name)}'


def get_sampler_scheduler(p, sampler):
    """Returns '{Sampler} {Scheduler}' / '{Scheduler}' / 'NOTHING_AND_SKIP_PREVIOUS_TEXT'"""
    if hasattr(p, 'scheduler') and hasattr(p, 'sampler_name'):
        if sampler:
            sampler_scheduler = get_sampler_scheduler_str(p.sampler_name, p.scheduler)
        else:
            sampler_scheduler = get_scheduler_str(p.sampler_name, p.scheduler)
        return sanitize_filename_part(sampler_scheduler, replace_spaces=False)
    return NOTHING_AND_SKIP_PREVIOUS_TEXT


class FilenameGenerator:
    replacements = {
        'basename': lambda self: self.basename or 'img',
        'seed': lambda self: self.seed if self.seed is not None else '',
        'seed_first': lambda self: self.seed if self.p.batch_size == 1 else self.p.all_seeds[0],
        'seed_last': lambda self: NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.batch_size == 1 else self.p.all_seeds[-1],
        'steps': lambda self:  self.p and self.p.steps,
        'cfg': lambda self: self.p and self.p.cfg_scale,
        'width': lambda self: self.image.width,
        'height': lambda self: self.image.height,
        'styles': lambda self: self.p and sanitize_filename_part(", ".join([style for style in self.p.styles if not style == "None"]) or "None", replace_spaces=False),
        'sampler': lambda self: self.p and sanitize_filename_part(self.p.sampler_name, replace_spaces=False),
        'sampler_scheduler': lambda self: self.p and get_sampler_scheduler(self.p, True),
        'scheduler': lambda self: self.p and get_sampler_scheduler(self.p, False),
        'model_hash': lambda self: getattr(self.p, "sd_model_hash", shared.sd_model.sd_model_hash),
        'model_name': lambda self: sanitize_filename_part(shared.sd_model.sd_checkpoint_info.name_for_extra, replace_spaces=False),
        'date': lambda self: datetime.datetime.now().strftime('%Y-%m-%d'),
        'datetime': lambda self, *args: self.datetime(*args),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format><Time Zone>]
        'job_timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),
        'prompt_hash': lambda self, *args: self.string_hash(self.prompt, *args),
        'negative_prompt_hash': lambda self, *args: self.string_hash(self.p.negative_prompt, *args),
        'full_prompt_hash': lambda self, *args: self.string_hash(f"{self.p.prompt} {self.p.negative_prompt}", *args),  # a space in between to create a unique string
        'prompt': lambda self: sanitize_filename_part(self.prompt),
        'prompt_no_styles': lambda self: self.prompt_no_style(),
        'prompt_spaces': lambda self: sanitize_filename_part(self.prompt, replace_spaces=False),
        'prompt_words': lambda self: self.prompt_words(),
        'batch_number': lambda self: NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.batch_size == 1 or self.zip else self.p.batch_index + 1,
        'batch_size': lambda self: self.p.batch_size,
        'generation_number': lambda self: NOTHING_AND_SKIP_PREVIOUS_TEXT if (self.p.n_iter == 1 and self.p.batch_size == 1) or self.zip else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'hasprompt': lambda self, *args: self.hasprompt(*args),  # accepts formats:[hasprompt<prompt1|default><prompt2>..]
        'clip_skip': lambda self: opts.data["CLIP_stop_at_last_layers"],
        'denoising': lambda self: self.p.denoising_strength if self.p and self.p.denoising_strength else NOTHING_AND_SKIP_PREVIOUS_TEXT,
        'user': lambda self: self.p.user,
        'vae_filename': lambda self: self.get_vae_filename(),
        'none': lambda self: '',  # Overrides the default, so you can get just the sequence number
        'image_hash': lambda self, *args: self.image_hash(*args)  # accepts formats: [image_hash<length>] default full hash
    }
    default_time_format = '%Y%m%d%H%M%S'

    def __init__(self, p, seed, prompt, image, zip=False, basename=""):
        self.p = p
        self.seed = seed
        self.prompt = prompt
        self.image = image
        self.zip = zip
        self.basename = basename

    def get_vae_filename(self):
        """Get the name of the VAE file."""

        import modules.sd_vae as sd_vae

        if sd_vae.loaded_vae_file is None:
            return "NoneType"

        file_name = os.path.basename(sd_vae.loaded_vae_file)
        split_file_name = file_name.split('.')
        if len(split_file_name) > 1 and split_file_name[0] == '':
            return split_file_name[1]  # if the first character of the filename is "." then [1] is obtained.
        else:
            return split_file_name[0]


    def hasprompt(self, *args):
        lower = self.prompt.lower()
        if self.p is None or self.prompt is None:
            return None
        outres = ""
        for arg in args:
            if arg != "":
                division = arg.split("|")
                expected = division[0].lower()
                default = division[1] if len(division) > 1 else ""
                if lower.find(expected) >= 0:
                    outres = f'{outres}{expected}'
                else:
                    outres = outres if default == "" else f'{outres}{default}'
        return sanitize_filename_part(outres)

    def prompt_no_style(self):
        if self.p is None or self.prompt is None:
            return None

        prompt_no_style = self.prompt
        for style in shared.prompt_styles.get_style_prompts(self.p.styles):
            if style:
                for part in style.split("{prompt}"):
                    prompt_no_style = prompt_no_style.replace(part, "").replace(", ,", ",").strip().strip(',')

                prompt_no_style = prompt_no_style.replace(style, "").strip().strip(',').strip()

        return sanitize_filename_part(prompt_no_style, replace_spaces=False)

    def prompt_words(self):
        words = [x for x in re_nonletters.split(self.prompt or "") if x]
        if len(words) == 0:
            words = ["empty"]
        return sanitize_filename_part(" ".join(words[0:opts.directories_max_prompt_words]), replace_spaces=False)

    def datetime(self, *args):
        time_datetime = datetime.datetime.now()

        time_format = args[0] if (args and args[0] != "") else self.default_time_format
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError:
            time_zone = None

        time_zone_time = time_datetime.astimezone(time_zone)
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError):
            formatted_time = time_zone_time.strftime(self.default_time_format)

        return sanitize_filename_part(formatted_time, replace_spaces=False)

    def image_hash(self, *args):
        length = int(args[0]) if (args and args[0] != "") else None
        return hashlib.sha256(self.image.tobytes()).hexdigest()[0:length]

    def string_hash(self, text, *args):
        length = int(args[0]) if (args and args[0] != "") else 8
        return hashlib.sha256(text.encode()).hexdigest()[0:length]

    def apply(self, x):
        res = ''

        for m in re_pattern.finditer(x):
            text, pattern = m.groups()

            if pattern is None:
                res += text
                continue

            pattern_args = []
            while True:
                m = re_pattern_arg.match(pattern)
                if m is None:
                    break

                pattern, arg = m.groups()
                pattern_args.insert(0, arg)

            fun = self.replacements.get(pattern.lower())
            if fun is not None:
                try:
                    replacement = fun(self, *pattern_args)
                except Exception:
                    replacement = None
                    errors.report(f"Error adding [{pattern}] to filename", exc_info=True)

                if replacement == NOTHING_AND_SKIP_PREVIOUS_TEXT:
                    continue
                elif replacement is not None:
                    res += text + str(replacement)
                    continue

            res += f'{text}[{pattern}]'

        return res


def get_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.

    The sequence starts at 0.
    """
    result = -1
    if basename != '':
        basename = f"{basename}-"

    prefix_length = len(basename)
    for p in os.listdir(path):
        if p.startswith(basename):
            parts = os.path.splitext(p[prefix_length:])[0].split('-')  # splits the filename (removing the basename first if one is defined, so the sequence number is always the first element)
            try:
                result = max(int(parts[0]), result)
            except ValueError:
                pass

    return result + 1


def save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name='parameters'):
    """
    Saves image to filename, including geninfo as text information for generation info.
    For PNG images, geninfo is added to existing pnginfo dictionary using the pnginfo_section_name argument as key.
    For JPG images, there's no dictionary and geninfo just replaces the EXIF description.
    """

    if extension is None:
        extension = os.path.splitext(filename)[1]

    image_format = Image.registered_extensions()[extension]

    if extension.lower() == '.png':
        existing_pnginfo = existing_pnginfo or {}
        if opts.enable_pnginfo:
            existing_pnginfo[pnginfo_section_name] = geninfo

        if opts.enable_pnginfo:
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in (existing_pnginfo or {}).items():
                pnginfo_data.add_text(k, str(v))
        else:
            pnginfo_data = None

        image.save(filename, format=image_format, quality=opts.jpeg_quality, pnginfo=pnginfo_data)

    elif extension.lower() in (".jpg", ".jpeg", ".webp"):
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB" if extension.lower() == ".webp" else "L")

        image.save(filename, format=image_format, quality=opts.jpeg_quality, lossless=opts.webp_lossless)

        if opts.enable_pnginfo and geninfo is not None:
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode")
                },
            })

            piexif.insert(exif_bytes, filename)
    elif extension.lower() == '.avif':
        if opts.enable_pnginfo and geninfo is not None:
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode")
                },
            })
        else:
            exif_bytes = None

        image.save(filename,format=image_format, quality=opts.jpeg_quality, exif=exif_bytes, subsampling='4:4:4')
    elif extension.lower() == ".gif":
        image.save(filename, format=image_format, comment=geninfo)
    else:
        image.save(filename, format=image_format, quality=opts.jpeg_quality)


def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
    """Save an image.

    Args:
        image (`PIL.Image`):
            The image to be saved.
        path (`str`):
            The directory to save the image. Note, the option `save_to_dirs` will make the image to be saved into a sub directory.
        basename (`str`):
            The base filename which will be applied to `filename pattern`.
        seed, prompt, short_filename,
        extension (`str`):
            Image file extension, default is `png`.
        pngsectionname (`str`):
            Specify the name of the section which `info` will be saved in.
        info (`str` or `PngImagePlugin.iTXt`):
            PNG info chunks.
        existing_info (`dict`):
            Additional PNG info. `existing_info == {pngsectionname: info, ...}`
        no_prompt:
            TODO I don't know its meaning.
        p (`StableDiffusionProcessing`)
        forced_filename (`str`):
            If specified, `basename` and filename pattern will be ignored.
        save_to_dirs (bool):
            If true, the image will be saved into a subdirectory of `path`.

    Returns: (fullfn, txt_fullfn)
        fullfn (`str`):
            The full path of the saved imaged.
        txt_fullfn (`str` or None):
            If a text file is saved for this image, this will be its full path. Otherwise None.
    """
    namegen = FilenameGenerator(p, seed, prompt, image, basename=basename)

    # WebP and JPG formats have maximum dimension limits of 16383 and 65535 respectively. switch to PNG which has a much higher limit
    if (image.height > 65535 or image.width > 65535) and extension.lower() in ("jpg", "jpeg") or (image.height > 16383 or image.width > 16383) and extension.lower() == "webp":
        print('Image dimensions too large; saving as PNG')
        extension = "png"

    if save_to_dirs is None:
        save_to_dirs = (grid and opts.grid_save_to_dirs) or (not grid and opts.save_to_dirs and not no_prompt)

    if save_to_dirs:
        dirname = namegen.apply(opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if forced_filename is None:
        if short_filename or seed is None:
            file_decoration = ""
        elif opts.save_to_dirs:
            file_decoration = opts.samples_filename_pattern or "[seed]"
        else:
            file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

        file_decoration = namegen.apply(file_decoration) + suffix

        add_number = opts.save_images_add_number or file_decoration == ''

        if file_decoration != "" and add_number:
            file_decoration = f"-{file_decoration}"

        if add_number:
            basecount = get_next_sequence_number(path, basename)
            fullfn = None
            for i in range(500):
                fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
                fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
                if not os.path.exists(fullfn):
                    break
        else:
            fullfn = os.path.join(path, f"{file_decoration}.{extension}")
    else:
        fullfn = os.path.join(path, f"{forced_filename}.{extension}")

    pnginfo = existing_info or {}
    if info is not None:
        pnginfo[pnginfo_section_name] = info

    params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    script_callbacks.before_image_saved_callback(params)

    image = params.image
    fullfn = params.filename
    info = params.pnginfo.get(pnginfo_section_name, None)

    def _atomically_save_image(image_to_save, filename_without_extension, extension):
        """
        save image with .tmp extension to avoid race condition when another process detects new image in the directory
        """
        temp_file_path = f"{filename_without_extension}.tmp"

        save_image_with_geninfo(image_to_save, info, temp_file_path, extension, existing_pnginfo=params.pnginfo, pnginfo_section_name=pnginfo_section_name)

        filename = filename_without_extension + extension
        if shared.opts.save_images_replace_action != "Replace":
            n = 0
            while os.path.exists(filename):
                n += 1
                filename = f"{filename_without_extension}-{n}{extension}"
        os.replace(temp_file_path, filename)

    fullfn_without_extension, extension = os.path.splitext(params.filename)
    if hasattr(os, 'statvfs'):
        max_name_len = os.statvfs(path).f_namemax
        fullfn_without_extension = fullfn_without_extension[:max_name_len - max(4, len(extension))]
        params.filename = fullfn_without_extension + extension
        fullfn = params.filename
    _atomically_save_image(image, fullfn_without_extension, extension)

    image.already_saved_as = fullfn

    oversize = image.width > opts.target_side_length or image.height > opts.target_side_length
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > opts.img_downscale_threshold * 1024 * 1024):
        ratio = image.width / image.height
        resize_to = None
        if oversize and ratio > 1:
            resize_to = round(opts.target_side_length), round(image.height * opts.target_side_length / image.width)
        elif oversize:
            resize_to = round(image.width * opts.target_side_length / image.height), round(opts.target_side_length)

        if resize_to is not None:
            try:
                # Resizing image with LANCZOS could throw an exception if e.g. image mode is I;16
                image = image.resize(resize_to, LANCZOS)
            except Exception:
                image = image.resize(resize_to)
        try:
            _atomically_save_image(image, fullfn_without_extension, ".jpg")
        except Exception as e:
            errors.display(e, "saving image as downscaled JPG")

    if opts.save_txt and info is not None:
        txt_fullfn = f"{fullfn_without_extension}.txt"
        with open(txt_fullfn, "w", encoding="utf8") as file:
            file.write(f"{info}\n")
    else:
        txt_fullfn = None

    script_callbacks.image_saved_callback(params)

    return fullfn, txt_fullfn


IGNORED_INFO_KEYS = {
    'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
    'loop', 'background', 'timestamp', 'duration', 'progressive', 'progression',
    'icc_profile', 'chromaticity', 'photoshop',
}


def read_info_from_image(image: Image.Image) -> tuple[str | None, dict]:
    items = (image.info or {}).copy()

    geninfo = items.pop('parameters', None)

    if "exif" in items:
        exif_data = items["exif"]
        try:
            exif = piexif.load(exif_data)
        except OSError:
            # memory / exif was not valid so piexif tried to read from a file
            exif = None
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        if exif_comment:
            geninfo = exif_comment
    elif "comment" in items: # for gif
        if isinstance(items["comment"], bytes):
            geninfo = items["comment"].decode('utf8', errors="ignore")
        else:
            geninfo = items["comment"]

    for field in IGNORED_INFO_KEYS:
        items.pop(field, None)

    if items.get("Software", None) == "NovelAI":
        try:
            json_info = json.loads(items["Comment"])
            sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")

            geninfo = f"""{items["Description"]}
Negative prompt: {json_info["uc"]}
Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
        except Exception:
            errors.report("Error parsing NovelAI image generation parameters", exc_info=True)

    return geninfo, items


def image_data(data):
    import gradio as gr

    try:
        image = read(io.BytesIO(data))
        textinfo, _ = read_info_from_image(image)
        return textinfo, None
    except Exception:
        pass

    try:
        text = data.decode('utf8')
        assert len(text) < 10000
        return text, None

    except Exception:
        pass

    return gr.update(), None


def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""

    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background

    return img.convert('RGB')


def read(fp, **kwargs):
    image = Image.open(fp, **kwargs)
    image = fix_image(image)

    return image


def fix_image(image: Image.Image):
    if image is None:
        return None

    try:
        image = ImageOps.exif_transpose(image)
        image = fix_png_transparency(image)
    except Exception:
        pass

    return image


def fix_png_transparency(image: Image.Image):
    if image.mode not in ("RGB", "P") or not isinstance(image.info.get("transparency"), bytes):
        return image

    image = image.convert("RGBA")
    return image
