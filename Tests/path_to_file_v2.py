import os

dirName = os.getcwd()
print(type(dirName), dirName)
dirName = os.path.join(dirName, '\images\line')
names = os.listdir(dirName)
for name in names:
    fullname = os.path.join(dirName, name)  # получаем полное имя
    if os.path.isfile(fullname):
        print(fullname)
