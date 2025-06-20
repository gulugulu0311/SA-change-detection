import os
def delete_files_in_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        print(f"文件夹 {folder_path} 下的所有文件已删除。")
    except Exception as e:
        print(f"删除文件时出错: {e}")
        
folders = ['TSSCD_TransEncoder', 'TSSCD_Unet', 'TSSCD_FCN']
# folders = ['TSSCD_TransEncoder']

model_data_folders = [f'model_data\\{model_name}' for model_name in folders]
for folder in model_data_folders:
    delete_files_in_folder(folder)
    
log_data_folders = [f'model_data\\log\\{log}' for log in folders]
for folder in log_data_folders:
    delete_files_in_folder(folder)
    

