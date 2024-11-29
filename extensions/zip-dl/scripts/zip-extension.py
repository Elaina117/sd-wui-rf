import os
import zipfile
import gradio as gr
import modules.scripts as scripts
import modules.script_callbacks as script_callbacks

def zip_with_progress(folder_path, zip_filename=None):
    """
    進捗バー付きでフォルダをZip化する関数
    """
    if not os.path.exists(folder_path):
        raise gr.Error(f"フォルダが見つかりません: {folder_path}")

    # Zipファイル名の決定
    if not zip_filename:
        zip_filename = os.path.basename(folder_path)
    
    if not zip_filename.lower().endswith('.zip'):
        zip_filename += '.zip'

    # Zipファイルの保存パス
    extensions_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_dir = os.path.join(extensions_dir, 'folder-zipper', 'downloads')
    os.makedirs(download_dir, exist_ok=True)
    
    output_path = os.path.join(download_dir, zip_filename)

    # ファイルリストと総サイズを取得
    files_to_zip = []
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_to_zip.append(file_path)
            total_size += os.path.getsize(file_path)

    try:
        progress_value = 0
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            processed_size = 0
            for file_path in files_to_zip:
                # 相対パスを計算
                archive_name = os.path.relpath(file_path, folder_path)
                
                # ファイルをZipに追加
                zipf.write(file_path, archive_name)
                
                # 進捗計算
                processed_size += os.path.getsize(file_path)
                progress_value = min(100, int((processed_size / total_size) * 100))

        return progress_value, output_path
    except Exception as e:
        raise gr.Error(f"Zipファイルの作成中にエラーが発生しました: {str(e)}")

def on_ui_tabs():
    """
    WebUIのタブとして表示するための関数
    """
    with gr.Blocks() as folder_zipper_interface:
        with gr.Row():
            folder_input = gr.Textbox(label="フォルダパス", placeholder="/content/output/txt2img")
            zip_name_input = gr.Textbox(label="Zipファイル名（オプション）", placeholder="未指定の場合はフォルダ名を使用")
        
        zip_button = gr.Button("フォルダをZip化")
        progress_bar = gr.Number(visible=False)  # Number型に戻す
        output_file = gr.File(label="作成されたZipファイル")

        zip_button.click(
            fn=zip_with_progress, 
            inputs=[folder_input, zip_name_input],
            outputs=[progress_bar, output_file]
        )

    return [(folder_zipper_interface, "ZIPでまとめてDL", "folder_zipper_tab")]

# 重要: スクリプトコールバックに追加
script_callbacks.on_ui_tabs(on_ui_tabs)