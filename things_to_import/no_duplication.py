l = []

f = open("list_of_missing_modules.txt", "r")

for i in f:
    if i not in l:
        l.append(i)
print(l)
f.close()
with open("no_duplication.txt", "w") as f:
    for i in l:
        f.write(i)