import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import os
from functools import partial
from time import sleep
from itertools import cycle
import matplotlib.colors as mcolors
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
#import matplotlib.cm


def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)

def write2text(chaser, data_dir, file_name, step, is_velocities=False):
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path, data_dir)

    #file_folder = f'vbar-vel{len(os.listdir(data_path))}'
    """
    data_dir : folder to store all velocities during all episodes

    file_name : text file containing all velocities for an episode
    """
    #file_path = os.path.join(data_path, file_folder)
    #file_path = os.path.join(file_path, file_name)
    file_path=os.path.join(data_path, file_name)
    #file name is changed in test loop
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
        """
        file_id = len(os.listdir(data_path))
        file_name = f'chaser{file_id}.txt'
        """
    if os.path.exists(file_path) == False:
        with open(file_path, 'w') as f:
            #f.write('0,0,0\n')
            pass

    state_arr = np.asarray(chaser.get_state_trace())
    print(f'state arr shape {state_arr.shape}')
    print('========================')
    print('state arr')
    print(state_arr)
    print('========================')

    if is_velocities:
        pos_arr = state_arr[step-1:, 3:6]
    else:
        pos_arr = state_arr[step-1:, 0:3]

    lines2write = ['\n'+str(pos[0])+','+str(pos[1])+','+str(pos[2]) for pos in pos_arr]

    print(f'WRITING TO {file_path}')
    print('----------------------')
    print('DATA WRITTING')
    print(lines2write)

    file = open(file_path, 'a')
    file.writelines(lines2write)
    file.close()


class render_visual:

    def __init__(self, data_dir = 'runs', only_inital = False, terminal_plots = False, run_file = None):
        style.use('default')
        matplotlib.use('WebAgg')
        #matplotlib.use('pgf')
        dir_path = os.getcwd()
        data_path = os.path.join(dir_path, data_dir)
        self.data_path = data_path
        self.only_inital = only_inital
        self.term_plots = terminal_plots
        self.dock_point = np.array([0, 60, 0])
        self.theta = 60.0
        self.center_termin = False
        self.run_file = run_file
        self.show_trajectories = True
        if os.path.exists(data_path) == False:
            os.makedirs(data_path)
        """
        file_id = len(os.listdir(data_path))
        file_name = f'chaser{file_id}.txt'
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'w') as f:
            #f.write('0,0,0')
            pass
        """


        self.fig = plt.figure(figsize = (5,5), dpi=100)
        self.ax = plt.axes(projection='3d')

        #plt.rcParams["figure.dpi"] = 1000


        #self.ax.set_zlim([-1200, 1200]) 
        #self.ax.set_xlim([-1200, 1200]) 
        #self.ax.set_ylim([-1200, 1200])
        self.ax.set_zlim([-800, 800]) 
        self.ax.set_xlim([-800, 800]) 
        self.ax.set_ylim([-100, 1200])
        """
        Plot target (cylinder) and LOS (cone)
        """
        x_center, y_center, radius, height = 0, 0, 60, 300
        Xc, Yc, Zc = self.data_for_cylinder_along_z(x_center, y_center, radius, height)

        if not only_inital:
            self.ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
        else:
            plot_terminal = input('Terminal visualization [y/n]: ')
            center_termin = input('center terminal plot? [y/n]')
            if plot_terminal == 'y':
                self.term_plots = True
                if center_termin == 'y':
                    self.center_termin = True
            else:
                pass
            print('only initial positions no target visualization')

        opt_los = input("Do you want to plot original LOS cone? (y/n): ")
        opt_view = input("Do you want view [1] or view [2] (1/2): ")
        opt_trajectory = input("Do you want to plot trajectories? (y/n): ")

        if opt_trajectory == 'y':
            self.show_trajectories = True
        elif opt_trajectory == 'n':
            self.show_trajectories = False
        else:
            raise ValueError('Invalid input')

        if opt_los == 'y':
            cone_length = 100
            print(f'plotting original LOS cone length {cone_length}')
            #X_los, Y_los, Z_los = self.data_for_cone_along_y(self.dock_point[0], self.dock_point[2], self.theta, self.dock_point[1], length=cone_length)
            X_los, Y_los, Z_los = self.parametric_cone_along_y(self.dock_point[0], self.dock_point[2], self.theta, self.dock_point[1], length=cone_length)
        else:
            cone_length = 800
            print(f'plotting LOS proposed with length {self.dock_point[1]}')
            #X_los, Y_los, Z_los = self.data_for_cone_along_y(self.dock_point[0], self.dock_point[2], self.theta, self.dock_point[1])
            X_los, Y_los, Z_los = self.parametric_cone_along_y(self.dock_point[0], self.dock_point[2], self.theta, self.dock_point[1], length=cone_length)

        #self.ax.plot_surface(X_los, Y_los, Z_los, alpha=0.5)
        x = np.linspace(-1.0, 1.0, len(X_los)**2)

        if not only_inital:
            #self.ax.scatter(X_los, Y_los, Z_los, s=0.1, c=x[::-1], cmap=cm.coolwarm, vmin=-1.0, vmax=1.0, alpha=0.8, antialiased=False, linewidths=0.8)
            #my_cmap = plt.get_cmap('coolwarm')
            cmap = cm.ScalarMappable(cmap='coolwarm_r')
            cmap.set_array(x[::-1])

            
            #self.ax.plot_surface(X_los, Y_los, Z_los, cmap=cmap.cmap, alpha=0.4, antialiased=False)
            face_colors = self.color_fade_with_y(X_los, Y_los, Z_los, cmap='coolwarm')
            #self.ax.contour3D(X_los, Y_los, Z_los, 100, cmap=my_cmap, linewidths=3, alpha=0.1)
        else:
            print('only initial positions no LOS cone')

        #X_slow, Y_slow, Z_slow = self.data_for_slowzone()
        #self.ax.plot_surface(X_slow, Y_slow, Z_slow, color='r', alpha = 0.25)


        #x=[0,0]
        #y=[60,860]
        #z=[0,0]

        #self.ax.plot3D(x, y, z, 'green')
        #self.ax.quiver(0, 60, 0, 0, 860, 0, headlength=1, headwidth=1, headaxislength=1, color=['black'], pivot='tail')
        #self.ax.view_init(elev=25, azim=60, roll=0) 

        if opt_view == '1':
            self.ax.view_init(elev=25, azim=60, roll=0)
        elif opt_view == '2':
            self.ax.view_init(elev=10, azim=50, roll=0)
        else:
            raise ValueError('Invalid view option')
        plt.draw()




    def color_fade_with_y(self, X, Y, Z, cmap='coolwarm'):
        # Create a figure and axes
        # Create the surface plot

        # Normalize Y to the range [0, 1] for color mapping
        y_normalized = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
        y_normalized = y_normalized * 2.0 - 1
        face_colors = plt.cm.coolwarm(y_normalized)
        surf = self.ax.plot_surface(X, Y, Z, cmap=cmap, facecolors=face_colors, rstride=5, cstride=5, linewidth=0.7, alpha=0.3, antialiased=False)
        #surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)

        # Obtain the face colors from the surface plot



        # Add a colorbar
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        # Return the modified face colors
        return face_colors

    def data_for_slowzone(self):
        u = np.linspace(0,2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        r = 500 #500m radius

        x_cen, y_cen, z_cen = self.dock_point[0], self.dock_point[1], self.dock_point[2]

        x = x_cen + (r * np.outer(np.cos(u), np.sin(v)) )
        y = y_cen + (r * np.outer(np.sin(u), np.sin(v)) )
        z = z_cen + (r * np.outer(np.ones(np.size(u)), np.cos(v)) )

        return x, y, z

    def data_for_cylinder_along_z(self, center_x, center_y, radius, height_z):
        #height_z
        z = np.linspace((height_z/2)*-1, height_z/2, 80)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        return x_grid,y_grid,z_grid

    def data_for_cone_along_y(self, center_x, center_z, theta, height_y, length = 800):
        max_y = height_y + length
        print(f'height_y {height_y} max_y {max_y}')
        print(f'center_x {center_x} center_z {center_z} theta {theta}')
        theta = 30.0

        h = length*(1/np.cos(theta))
        r = np.sqrt(h**2.0 - length**2.0)
        min_iterations = max_y - height_y
        Y = np.linspace(height_y-1.5, max_y, 10000)
        X = center_x + Y * np.sin(theta*Y)
        Z = center_z + Y * np.cos(theta*Y)
        return X, Y, Z

    def parametric_cone_along_y(self, center_x, center_z, theta, height_y, length = 800):
        h = length*(1/np.cos(theta))
        radius = length / np.sqrt(3)

        alpha = np.linspace(0, 2.*np.pi, 75)  # angle around the cone
        r = np.linspace(0, radius, 75)  # radius from the center of the cone
        Alpha, R = np.meshgrid(alpha, r)  # create a grid of (theta, r) points

        # Convert polar (cone) coordinates to cartesian coordinates
        X = R * np.cos(Alpha)
        Z = R * np.sin(Alpha)  # shifting the cone to have vertex at y = 60
        Y = np.tan(np.radians(theta)) * np.sqrt(X**2 + Z**2) + 60  # height of the cone from its vertex
        return X, Y, Z

    def animate(self, i, file_name = None):

        cycol = cycle('rgbycmk')

        for run_file in os.listdir(self.data_path):
            print(f'plotting {run_file}')
            data = os.path.join(self.data_path, run_file)

            if os.path.exists(data) == False:
                #file_path = os.path.join(data_path, file_name)
                with open(data, 'w') as f:
                    #f.write('0,0,0')
                    pass
            
            if not self.show_trajectories:
                break
            if self.run_file != None:
                data = os.path.join(self.data_path, self.run_file)

            with open(data) as f:
                lines = f.readlines()
            xs, ys, zs = np.array([]), np.array([]), np.array([])
            for idx, line in enumerate(lines):
                line = line.strip()
                if len(line) > 1:
                    x, y, z = line.split(',')
                    x, y, z = np.float64(x), np.float64(y), np.float64(z)
                    xs = np.append(xs, [x])
                    ys = np.append(ys, [y])
                    zs = np.append(zs, [z])
                if idx > 1 and self.only_inital and self.term_plots == False:
                    self.ax.set_xlim([-400, 400]) 
                    self.ax.set_ylim([400, 1200])
                    self.ax.set_zlim([-400, 400]) 
                    break
                elif self.term_plots == True:
                    #self.ax.lab
                    if self.center_termin != True:
                        self.ax.set_xlim([-400, 400]) 
                        self.ax.set_ylim([400, 1200])
                        self.ax.set_zlim([-400, 400]) 
                    else:
                        self.ax.set_xlim([-500, 500]) 
                        self.ax.set_ylim([-440, 560])
                        self.ax.set_zlim([-500, 500])

            if self.term_plots == False:
                self.ax.scatter(xs[1], ys[1], zs[1], marker='o', c='purple', s=1.5, alpha=0.99)
            #fade = np.linspace(-1.0, 1.0, len(xs))
            #self.ax.scatter(xs, ys, zs, marker='o', s=0.05, c=next(cycol), alpha=0.9)
            if self.term_plots == False:
                if self.run_file != None:
                    self.ax.plot(xs, ys, zs, c='g', linewidth=0.9, alpha=0.95, markevery=15, antialiased=False)
                else:
                    self.ax.plot(xs[:i], ys[:i], zs[:i], c=next(cycol), linewidth=0.7, alpha=0.5, markevery=15, antialiased=True)
            else:
                term_scatter = self.ax.scatter(xs[-1], ys[-1], zs[-1], marker='o', c='r', s=0.5)

                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x_sphere = 20 * np.outer(np.cos(u), np.sin(v))
                y_sphere = (20 * np.outer(np.sin(u), np.sin(v))) + 60
                z_zphere = 20 * np.outer(np.ones(np.size(u)), np.cos(v))

                dock_scatter = self.ax.scatter([0], [60], [0], marker='o', c='r', s=1.5)

                #legend1 = self.ax.legend(*term_scatter.legend_elements(), loc="upper right", title="State Legend")
                #self.ax.add_artist(legend1)
                if False:
                    legend_elements = [mplot3d.art3d.Line3D([0], [0], [0], marker='o', color='b', label='Final Position', markersize=8),
                                    mplot3d.art3d.Line3D([0], [0], [0], marker='o', color='r', label='Docking Port', markersize=8)]
                    self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                    #self.ax.plot_surface(x_sphere, y_sphere, z_zphere,  rstride=4, cstride=4, color='g', alpha=0.2)
            if self.run_file != None:
                self.ax.set_xlim([-400, 400]) 
                self.ax.set_ylim([-200, 1000])
                self.ax.set_zlim([-400, 400])
                break 
        self.ax.set_xlabel('X [m]', fontsize=10, labelpad=5)
        self.ax.set_ylabel('Y [m]', fontsize=10, labelpad=5)
        self.ax.set_zlabel('Z [m]', fontsize=10, labelpad=5)
        #self.ax.legend()
        if self.center_termin == True:
            self.ax.set_xticks(np.arange(-500, 501, 120))
            self.ax.set_yticks(np.arange(-440, 561, 120))
            self.ax.set_zticks(np.arange(-500, 501, 120))
        # Set tick font size
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels() + self.ax.get_zticklabels()):
            label.set_fontsize(8)


        #self.ax.legend()
        #self.ax.tick_params(axis='both', which='major', labelsize=5)
        #self.ax.tick_params(axis='both', which='minor', labelsize=5)
        plt.draw()
        sleep(1.0)

        if i > 125:
            return

    def render_animation(self, file_name = None):
        """
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection='3d')
        ax.set_zlim([-1000, 1000]) 
        ax.set_xlim([-1000, 1000]) 
        ax.set_ylim([-1000, 1000])
        """
        #plt.ion()
        #data = os.path.join(self.data_path, file_name)
        ani = animation.FuncAnimation(self.fig, partial(self.animate, file_name=file_name), repeat=False, cache_frame_data=True, interval=100)
        #print(ani.to_jshtml())
        #plt.ion()
        plt.show(block=False)
        #plt.savefig('myfigure.eps', format='eps', dpi=300)
        #plt.show()
        #sleep(1.0)

    def save(self):
        self.fig.show()
        self.fig.savefig("1003dvbar.eps", format="eps",transparent=True)
       #plt.savefig('1003dvbar.png')
