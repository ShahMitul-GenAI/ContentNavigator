import os
import glob
from pathlib import Path
import shutil
#
# with open("./notifications/note1.txt", 'r') as fp:
#     print(fp.read())
#
# for i in range(1,5):
#     print("/conclusive/" + "note" + str(i) + ".txt")
#
#
# file_path = "C:/Users/Mast_Nijanand/customer_review_app/"
# check_notification = str(file_path) + "notifications/"
# i = 1
# print(str(check_notification) + ("note" + str(i) + ".txt"))
# print(os.path.isfile(str(check_notification) + ("note" + str(i) + ".txt")))
# with open((str(check_notification) + ("note" + str(i) + ".txt")), 'r') as fp:
#     lines = fp.read()
#     line = lines.splitlines()
#     for each in line:
#         st.write(each)

target_folder = folder_path = str(os.path.dirname(os.path.abspath(__file__))) + "/docs"
print(target_folder)

# clearing previously loaded pool of docs
folder_path = str(target_folder)

[f.unlink() for f in Path(str(target_folder)).iterdir() if f.is_file()]
# list( map( os.unlink, (os.path.join(target_folder,f) for f in os.listdir(target_folder)) ) )

# shutil.rmtree(folder_path)
# os.mkdir(folder_path)

print(target_folder)
# files = glob.glob(folder_path)
# for f in files:
#     os.remove(f)
