import matplotlib.pyplot as plt
import glob
import os
import numpy as np

x = []
y = []
z = []
original_dir = os.getcwd()

#option = input('options (pos) or (vel) plot')

#isvalid_opt = False

while True:
    option = input('options (pos), (vel), or (ctr) plot : ')
    if option == 'pos':
        #isvalid_opt == True
        os.chdir('runs/vbar0')
        break
    elif option == 'vel':
        #isvalid_opt == True
        os.chdir('velocities/vbar0')
        break
    elif option == 'ctr':
        os.chdir('actuations/vbar0')
        break
    else:
        print(f'INVALID INPUT {option} WAS ENTERED PLEASE CHOOSE EITHER pos vel or ctr AS INPUT')

print(os.getcwd())

filename = input('name of run file (chaser(n).txt)')
with open(filename, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
    
        if idx <= 0:
            pass
        else:
            data = line.split(',')
            print(data)
            x.append(float(data[0]))
            y.append(float(data[1]))
            z.append(float(data[2]))

plt.xlim(0, 1000)

if option == 'pos':
    plt.ylim(-400, 1200)
    plt.plot(x, label='x')
    plt.plot(y, label='y')
    plt.plot(z, label='z')
    plt.ylabel("Position (m)")
elif option == 'vel':
    plt.ylim(-5, 5)
    plt.plot(x, label='xdot')
    plt.plot(y, label='ydot')
    plt.plot(z, label='zdot')
    plt.ylabel("Velocity (m/s)")
else:
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    spacessss = np.linspace(0,1000, len(x[::100]))
    plt.ylim(-10, 10)
    
    plt.scatter(spacessss, x[::100], label='ux')
    plt.scatter(spacessss, y[::100], label='uy')
    plt.scatter(spacessss, z[::100], label='uz')
    """
    plt.scatter(x[::10], spacessss, label='ux')
    plt.scatter(y[::10], spacessss, label='uy')
    plt.scatter(z[::10], spacessss, label='uz')
    """
    plt.ylabel("Newtons (N)")

plt.xlabel("Time (seconds)")
#plt.ylabel("Position (m)")

#plt.xlabel('Data points')
#plt.ylabel('Data values')

if option == 'pos':
    eps_fname = 'posplotsingle'
    plt.title('Sample of Single Trajectory Position Components')
elif option == 'vel':
    eps_fname = 'velplotsingle'
    plt.title('Sample of Single Trajectory Velocity Components')
else:
    eps_fname = 'actuationplotsingle'
    plt.title('Sample of Single Trajectory Actuations')

plt.legend()
plt.grid(True)
os.chdir(original_dir)
plt.savefig(f'{eps_fname}.eps', format='eps')

plt.show()
