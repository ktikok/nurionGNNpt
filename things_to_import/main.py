# import check_uninstalled_packages
import os

# if __name__ == "__main__":
cmd = 'nohup python check_uninstalled_packages.py & > nohup.out ;tail -f nohup.out'
# print(cmd)
os.system(cmd)