import torch
from safetensors.torch import safe_open
from modules import scripts, sd_models, shared
import gradio as gr
from modules.processing import process_images


class KeyBasedModelMerger(scripts.Script):
    def title(self):
        return "Key-based model merging"

    def ui(self, is_txt2img):
        # UI コンポーネントを定義
        model_names = sorted(sd_models.checkpoints_list.keys(), key=str.casefold)
        
        model_a_dropdown = gr.Dropdown(
            label="Model A", choices=model_names, value=model_names[0] if model_names else None
        )
        model_b_dropdown = gr.Dropdown(
            label="Model B", choices=model_names, value=model_names[0] if model_names else None
        )
        keys_and_alphas_textbox = gr.Textbox(
            label="マージするテンソルのキーとマージ比率 (部分一致, 1行に1つ, カンマ区切り)",
            lines=5,
            placeholder="例:\nmodel.diffusion_model.input_blocks.0,0.5\nmodel.diffusion_model.middle_block,0.3"
        )
        merge_checkbox = gr.Checkbox(label="モデルのマージを有効にする", value=True)
        use_gpu_checkbox = gr.Checkbox(label="GPUを使用", value=True)  # GPU/CPU切り替えチェックボックス
        batch_size_slider = gr.Slider(minimum=1, maximum=500, step=1, value=250, label="KeyMgerge_BatchSize")

        return [model_a_dropdown, model_b_dropdown, keys_and_alphas_textbox, merge_checkbox, use_gpu_checkbox, batch_size_slider]

    def run(self, p, model_a_name, model_b_name, keys_and_alphas_str, merge_enabled, use_gpu, batch_size):
        if not model_a_name or not model_b_name:
            print("Error: Model A or Model B is not selected.")
            return p

        try:
            model_a_filename = model_a_name if os.path.isfile(model_a_name) else None
            model_b_filename = model_b_name if os.path.isfile(model_b_name) else None
        except KeyError as e:
            print(f"Error: Selected model is not found in checkpoints list. {e}")
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

            # デバイスの設定 (GPUかCPUか選べるようにする)
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

        # 必要に応じて process_images を実行
        return process_images(p)
