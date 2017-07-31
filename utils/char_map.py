
# From Baidu ba-dls-deepspeech - https://github.com/baidu-research/ba-dls-deepspeech

char_map_str = """
' 1
<SPACE> 2
a 3
b 4
c 5
d 6
e 7
f 8
g 9
h 10
i 11
j 12
k 13
l 14
m 15
n 16
o 17
p 18
q 19
r 20
s 21
t 22
u 23
v 24
w 25
x 26
y 27
z 28
"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[2] = ' '
