from copy import copy

k = 4#scale, k cells in 1 cm

#sizes in cm
r = 1 #Radius of ball
d_1 = 8#Distance between ball`s center and right plate 
d_2 = 4#===========same====================left plate
a = 20#Diameter of plate
a_thikness = 1#Thickness of plate
n = 200#Size of grid
phi_0 = 10000#Module of potential of plates
phi_1 = 10000#Potential of ball

def numer(x, y):
    return n*y + x
def positon(number):
    return (number%n, number//n)
def sign(i):
    if i == 0:
        return 0
    elif i > 0:
        return 1
    return -1

#grid keeps potential in one-dimentional array
border_conditions = [0 for i in range(0, n**2)]#flag for cells with known potential
mapa = [0 for i in range(0, n**2)]#potentials
new_mapa = [0 for i in range(0, n**2)]#next potentials distribution

#Initial conditions
for i in range(0, n**2):
    #1
    p = positon(i)
    if( (p[0] - n//2)**2 + (p[1] - n//2)**2 <= (k*r)**2 ):
        #ball
        mapa[i] = phi_1
        new_mapa[i] = phi_1
        border_conditions[i] = 1
    elif( p[1] >= n//2-k*a and p[1] <= n//2+k*a and p[0] >= n//2+k*d_1-a_thikness 
         and p[0] <= n//2+k*d_1+a_thikness ):
        #plate +
        mapa[i] = phi_0
        new_mapa[i] = phi_0
        border_conditions[i] = 1
    elif( p[1] >= n//2-k*a and p[1] <= n//2+k*a and
          p[0] >= n//2-k*d_2-a_thikness and p[0] <= n//2-k*d_2+a_thikness ):
        #plate -
        mapa[i] = -phi_0
        new_mapa[i] = -phi_0
        border_conditions[i] = 1
    elif( p[0] == 0 or p[0] == n-1 or p[1] == 0 or p[1] == n - 1 ):
        #border
        mapa[i] = 0
        new_mapa[i] = 0
        border_conditions[i] = 1
print("Initial conditions done")

#Iterations
for t in range(3000):
    for i in range(n**2):
        if border_conditions[i] == 1:
            continue
        p = positon(i)
        x = 0
        if p[1]-n//2 != 0:
            x = 1/(1/abs(p[1]-n//2)+4) * (mapa[i-1]+(1/abs(p[1]-n//2)+1)*mapa[i+sign(p[1]-n//2)*n]+mapa[i+1]+mapa[i-sign(p[1]-n//2)*n])
        else:
            x = 1/6 * (2*(mapa[i-n] + mapa[i+n]) + mapa[i-1] + mapa[i+1])
        new_mapa[i] = x
    mapa = copy(new_mapa)
print("Jacobi iterations done")

#Calculating charge
q = 0
for i in range(n//2-r*k - 3, n//2+r*k + 3):
    for j in range(n//2, n//2 +r*k  + 3):
        q += mapa[numer(i, j+1)] - mapa[numer(i, j)]
        q += abs(j-n//2) * (-4 * mapa[numer(i, j)] + mapa[numer(i+1, j)] +  mapa[numer(i-1, j)] + mapa[numer(i, j-1)] + mapa[numer(i, j+1)])
#Charge is proportional the length of cell size so we  divide by it 
print(-8.85*10**(-12)*3.1415*2*q / (k * 100))
print("Charge calculated")

#Drawing a diagram
print("Start creating a diagram")
from tkinter import Tk, Canvas, PhotoImage, mainloop

zoom = 4
WIDTH, HEIGHT = n*zoom, n*zoom

window = Tk()
canvas = Canvas(window, width=WIDTH, height=HEIGHT, bg="#000000")
canvas.pack()
img = PhotoImage(width=WIDTH, height=HEIGHT)
canvas.create_image((WIDTH/2, HEIGHT/2), image=img, state="normal")

for i in range(n**2):
    for k in range(zoom):
        for b in range(zoom):
            r = int((max(phi_0, phi_1)+mapa[i])*44//max(phi_0, phi_1))+10
            img.put(f'#2D{r}4A', (positon(i)[0]*zoom+k, positon(i)[1]*zoom+b))

mainloop()
