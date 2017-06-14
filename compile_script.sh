#cython3 --embed -o hello.c hello.py
#gcc -Os -I /usr/include/python3.3m -o hello hello.c -lpython3.3m -lpthread -lm -lutil -ldl


cython --embed -o $1.c $1.py
gcc -Os -I /usr/include/python2.7 -o $1 $1.c -lpython2.7 -lpthread -lm -lutil -ldl
