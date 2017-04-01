
import fileinput
import re
 
for line in fileinput.input(inplace=1, backup='.bak'):
    line = re.sub('\(','', line.rstrip())
    print(line)
