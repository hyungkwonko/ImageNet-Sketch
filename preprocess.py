from os import path

# load file
file1 = open('data/sketch/sketchPath.csv', 'r')
Lines = file1.readlines()

# file to save
file2 = open('sketchPath_process.csv', 'w')
file2.writelines(f'path,label\n')

count = 0
for line in Lines:
    count += 1
    tmp = line.strip()
    tmp1 = tmp[:24]
    tmp1 = tmp1.replace(',', '')

    # copy line if file exists
    yesno = path.isfile(f'data/sketch/{tmp1}')
    if yesno:
        file2.writelines(f'{tmp}\n')
file2.close()
