import os

def Makedir(path):
    folder = os.path.exists(path)
    if (not folder):
        os.makedirs(path)

root_folder = "/data/chenzhibo/data/0.35T1217select/0.35Tok/"
d2n = 'dcm2nii '
para = ' -d n -o '
output_path = '/data/chenzhibo/data/0.35T_ok_nii'

for i, subject_folder in enumerate(os.listdir(root_folder)):
    subject_folder_path = os.path.join(root_folder, subject_folder)
    if os.path.isdir(subject_folder_path):
        print(subject_folder[:8])
        for t in os.listdir(subject_folder_path):
            folder_path = os.path.join(subject_folder_path, t)
            # print(subject_folder[:8])
            if os.path.isdir(folder_path) and "T1" in t:
                cmd = d2n + para + '\"' + output_path + f'/{subject_folder[:8]}/{t}' + '\" ' + '\"' + folder_path + '\"'
                Makedir(output_path+f'/{subject_folder[:8]}/{t}')
                os.system(cmd)
        print(f'{subject_folder[:8]} finished')
    else:
        i -= 1
print('Done!')