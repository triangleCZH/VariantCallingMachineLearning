#this is to generate the text to auto build all new folders for each snapshot and each part
print("cp ../script/extract_csv.py .")
print("mkdir log")
for i in range (0,10):
  print("mkdir result0%d" % i)
  print("cd result0%d" % i)
  for j in range(1,10):
    print("mkdir s%d" % j)
    print("cd s%d" % j)
    print("cp ../../extract_csv.py extract%d%d.py" % (i, j))
    print("sed -i \"s/iiiii/%d/g; s/jjjjj/%d/g\" extract%d%d.py" % (i, j, i, j))
    print("cd ..")
  print("cd ..")
