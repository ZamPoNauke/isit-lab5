import os

path_to_current_file = os.path.realpath(__file__)
print('path to the current file:', path_to_current_file)

path_to_current_folder = os.path.dirname(path_to_current_file)
print('path to the current folder:', path_to_current_folder)

folder_inside_current_folder = os.path.dirname(os.path.join(path_to_current_folder, 'static/'))
print('path to the custom folder:', folder_inside_current_folder)

