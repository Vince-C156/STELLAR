import os
import matplotlib.pyplot as plt


#read from run folder


def graph100(runs_dir, graph_all=True):
    print(runs_dir)

    if graph_all != True:
        opt = str(input('Which component to plot in isolation [x] [y] [z]'))

    for filename in os.listdir(runs_dir):
        if filename.endswith('.txt'):
            f = open(f'{runs_dir}/{filename}', "r")
            pos_data = f.readlines()

            xs = []
            ys = []
            zs = []
            for idx, pos in enumerate(pos_data):
                pos_split = pos.split(',')

                if idx > 0:
                    x = float(pos_split[0])
                    y = float(pos_split[1])
                    z = float(pos_split[2])

                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

            f.close()

            # plt.plot(xs, label='x')
            # plt.plot(ys, label='y')
            # plt.plot(zs, label='z')
            if graph_all:
                l1, = plt.plot(xs, color = 'red')
                l2, = plt.plot(ys, color = 'green')
                l3, = plt.plot(zs, color = 'blue')
            else:
                if opt == 'x':
                    l1, = plt.plot(xs, color = 'red')
                elif opt =='y':
                    l2, = plt.plot(ys, color = 'green')
                elif opt == 'z':
                    l3, = plt.plot(zs, color = 'blue')
                else:
                    print(f'OPTION {opt} is invalid')
                    raise Exception
    plt.xlim(0, 450)

    if graph_all:
        lhandle = [l1, l2, l3]
        llabel = ['x', 'y', 'z']
    else:
        if opt == 'x':
            lhandle = [l1]
            if runs_dir == 'velocities/vbar0':
                llabel = ['xdot']
                plt.ylim(-5, 5)
            else:
                llabel = ['x']
        elif opt =='y':
            lhandle = [l2]
            if runs_dir == 'velocities/vbar0':
                llabel = ['ydot']
                plt.ylim(-5, 5)
            else:
                llabel = ['y']
        elif opt == 'z':
            lhandle = [l3]
            if runs_dir == 'velocities/vbar0':
                llabel = ['zdot']
                plt.ylim(-5, 5)
            else:
                llabel = ['z']
        else:
            print(f'OPTION {opt} is invalid')
            raise Exception
    plt.grid(visible=True)
    plt.legend(handles=lhandle, labels=llabel)
    #leg = plt.legend(loc='upper right')
    plt.xlabel("Time [s]", fontsize=15)
    plt.ylabel("Position [m]", fontsize=15)

    plt.savefig('100pos.eps', format='eps')
    plt.show()



usr_vel = str(input('Graph velocity? [Y] [N]'))

if usr_vel == 'Y':
    data_dir = 'velocities/vbar0'
else:
    data_dir = 'runs/vbar0'

usr_input = str(input('Graph all components? [Y] [N]'))

if usr_input == 'Y':
    graph100(data_dir)
elif usr_input == 'N':
    graph100(data_dir, False)
else:
    print(f'INVALID OPTION {usr_input}')
    raise Exception


