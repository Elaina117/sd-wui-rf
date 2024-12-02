import torch
from safetensors.torch import safe_open
from modules import scripts, sd_models, shared
import gradio as gr
from modules.processing import process_images


class KeyBasedModelMerger(scripts.Script):
    def title(self):
        # マージ済みフラグを追加
        self.has_merged = False
        return "モデルをマージ"

    def ui(self, is_txt2img):
        model_names = sorted(sd_models.checkpoints_list.keys(), key=str.casefold)
        
        model_a_dropdown = gr.Dropdown(label="モデル A", choices=model_names)
        model_b_dropdown = gr.Dropdown(label="モデル B", choices=model_names)
        
        merge_ratio_slider = gr.Slider(
            minimum=0, 
            maximum=1, 
            step=0.01, 
            value=0.5, 
            label="マージ比率（0.0で「モデルA」、1.0で「モデルB」になります）"
        )
        
        keys_and_alphas_textbox = gr.Textbox(
            label="マージするテンソルのキーとマージ比率 (部分一致, 1行に1つ, カンマ区切り)",
            lines=5,
            placeholder="ここにキーとマージ比率を入力"
        )
        
        autofill_button = gr.Button("比率を挿入")
        
        merge_checkbox = gr.Checkbox(label="モデルのマージを有効にする", value=True)
        use_gpu_checkbox = gr.Checkbox(label="GPUを使用", value=True)
        batch_size_slider = gr.Slider(minimum=1, maximum=500, step=1, value=250, label="KeyMerge_BatchSize")
        
        # モデルAドロップダウンが変更されたら
        model_a_dropdown.change(
            fn=lambda: setattr(self, 'has_merged', False),
            inputs=[],
            outputs=[]
        )

        # モデルBドロップダウンが変更されたら
        model_b_dropdown.change(
            fn=lambda: setattr(self, 'has_merged', False),
            inputs=[],
            outputs=[]
        )
        # マージの内容が変更されたら
        keys_and_alphas_textbox.change(
            fn=lambda: setattr(self, 'has_merged', False),
            inputs=[],
            outputs=[]
        )
        
        autofill_button.click(
            fn=lambda ratio: f"""model.diffusion_model.input_blocks.0,{ratio}
model.diffusion_model.input_blocks.1,{ratio}
model.diffusion_model.input_blocks.2,{ratio}
model.diffusion_model.middle_block,{ratio}
model.diffusion_model.output_blocks.0,{ratio}
model.diffusion_model.output_blocks.1,{ratio}
model.diffusion_model.output_blocks.2,{ratio}""",
            inputs=[merge_ratio_slider],
            outputs=[keys_and_alphas_textbox],
        )

        return [
            model_a_dropdown, model_b_dropdown, 
            keys_and_alphas_textbox, 
            merge_checkbox, 
            use_gpu_checkbox, 
            batch_size_slider
        ]

    def run(self, p, model_a_name, model_b_name, keys_and_alphas_str, merge_enabled, use_gpu, batch_size):
        # すでにマージ済みの場合は処理をスキップ
        if self.has_merged:
            return process_images(p)

        if not model_a_name or not model_b_name:
            print("エラー： モデルAまたはモデルBが選択されていません。")
            return p

        try:
            model_a_filename = sd_models.checkpoints_list[model_a_name].filename
            model_b_filename = sd_models.checkpoints_list[model_b_name].filename
        except KeyError as e:
            print(f"エラー： Selected model is not found in checkpoints list. {e}")
            return p

        # マージ処理
        if merge_enabled:
            input_keys_and_alphas = []
            for line in keys_and_alphas_str.split("\n"):
                if "," in line:
                    key_part, alpha_str = line.split(",", 1)
                    try:
                        alpha = float(alpha_str)
                        input_keys_and_alphas.append((key_part, alpha))
                    except ValueError:
                        print(f"Invalid alpha value in line '{line}', skipping...")
            
            # state_dictからキーのリストを事前に作成
            model_keys = list(shared.sd_model.state_dict().keys())
            
            # 部分一致検索を行う
            final_keys_and_alphas = {}
            for key_part, alpha in input_keys_and_alphas:
                for model_key in model_keys:
                    if key_part in model_key:
                        final_keys_and_alphas[model_key] = alpha

            # デバイスの設定
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

            # バッチ処理でキーをまとめて処理
            batched_keys = list(final_keys_and_alphas.items())

            # モデルAとモデルBからテンソルをまとめて取得
            with safe_open(model_a_filename, framework="pt", device=device) as f_a, \
                 safe_open(model_b_filename, framework="pt", device=device) as f_b:

                # バッチごとに処理
                for i in range(0, len(batched_keys), batch_size):
                    batch = batched_keys[i:i + batch_size]

                    # バッチでテンソルを取得して一度にマージ
                    tensors_a = [f_a.get_tensor(key) for key, _ in batch]
                    tensors_b = [f_b.get_tensor(key) for key, _ in batch]
                    alphas = [final_keys_and_alphas[key] for key, _ in batch]

                    # バッチでテンソルをマージして一度に適用
                    for key, alpha, tensor_a, tensor_b in zip([key for key, _ in batch], alphas, tensors_a, tensors_b):
                        # 直接 state_dict にマージ結果を適用
                        shared.sd_model.state_dict()[key].copy_(torch.lerp(tensor_a, tensor_b, alpha).to(device))
                        print(f"merged {alpha}:{key}")

            # マージ済みフラグを設定
            self.has_merged = True

        # 必要に応じて process_images を実行
        return process_images(p)