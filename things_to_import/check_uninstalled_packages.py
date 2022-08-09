import os
import time
# print("you can use checkAndWrite()")
# def checkAndWrite():
f = open("list_of_import.txt", "r")
# print(f.readline())
# print(type(f))
# print(len(f))

os.system('echo "" > list_of_missing_modules.txt')
# cmd = 'ls -l'
# os.system(cmd)

# for j in range(10):
check = []
for i in f:
    if i not in check:
        check.append(i)
	# print(i)
	# print(type(i))
        cmd = 'nohup python -c "' + i[:-1] + '" >>  list_of_missing_modules.txt &'
    	# print(cmd)
        os.system(cmd)
f.close()
    # os.system('python -c '+i)
    # break
# import subprocess
# out = subprocess.Popen(['ps', 'aux', '|', 'grep', '22a06'], 
#            stdout=subprocess.PIPE, 
#            stderr=subprocess.STDOUT)
# print(out)
# print(type(out))
# stdout,stderr = out.communicate()
# print(stdout)
# print(stderr)

flag = 1
t = 0
print('Waiting for procssing')
while(flag):
    os.system("ps aux|grep 22a06 > process_check.txt")
    f = open("process_check.txt", "r")
    flag = 0
    for i in f:
        # print(i)
        if "python -c " in i:

            flag = 1
            # print('not done yet------------------------')

    f.close()
    time.sleep(1)
    # print('.',end='')
    print('.')
print('')

