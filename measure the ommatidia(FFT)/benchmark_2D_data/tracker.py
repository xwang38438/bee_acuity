import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import Arrow, Circle, Polygon, Rectangle
import matplotlib
from matplotlib import animation
import PIL
import os


class tracker_window():

    def __init__(self, dirname="./", num_markers=6, fn='data.npy'):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0

        # markers and data file
        self.num_markers = num_markers
        self.range_markers = np.arange(self.num_markers)
        self.curr_marker = 0

        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                  (0, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0)]
        self.colors = []
        for x in self.range_markers:
            self.colors += [colors[x % len(colors)]]

        self.marker_view = 0
        self.fn = os.path.join(self.dirname, fn)
        if os.path.isfile(self.fn):
            self.markers = np.load(self.fn)
            # remove markers for files that have been deleted
            if os.path.isfile(os.path.join(self.dirname, 'order.npy')):
                self.old_order = np.load(
                    os.path.join(self.dirname, 'order.npy'))
                self.old_order.sort()
                keepers = np.array([fn in self.filenames for fn in self.old_order])
                self.old_order = self.old_order[keepers]
                self.markers = self.markers[:, keepers]
            else:
                self.old_order = np.array([])
            if self.markers.shape[1] < self.num_frames:
                old_files = [fn in self.old_order for fn in
                             self.filenames]
                z = np.zeros((self.num_markers, self.num_frames, 2))
                z[:, old_files] = self.markers
                self.markers = z
        else:
            self.markers = np.zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1
            self.imarkers = np.zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1

        np.save(os.path.join(self.dirname, "order.npy"), self.filenames)
        self.data_changed = False

        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 8
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height

        # self.figure = plt.figure(1, figsize=(
        #     figsize[0]+1, figsize[1]+2), dpi=90)
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        # xmarg, ymarg = .2, .1
        fig_left, fig_bottom, fig_width, fig_height = .15, .1, .75, .85
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='gray')
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        for i in range(self.num_markers):
            plt.plot([-1], [-1], 'o', mfc=self.colors[i])
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)

        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider ing frames
        # axframe = plt.axes([0.5-.65/2, 0.04, 0.65, 0.02])
        axframe = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)

        # radio buttons for marker selection
        self.radioframe = plt.axes([0.04, .60, 0.10, 0.20], frameon=False)
        self.radioframe.set_title("marker", ha='right')
        labs = [str(x + 1) for x in range(self.num_markers)]
        labs = tuple(labs)
        self.radiobuttons = RadioButtons(
            self.radioframe, labs, activecolor='k')
        # color the buttons
        for i in np.arange(len(labs)):
            self.radiobuttons.labels[i].set_color(self.colors[i])
        self.radiobuttons.on_clicked(self.marker_button)

        # connect some keys
        self.cidk = self.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

        # make a list of objects and filenames to save
        self.objects_to_save = {}
        self.objects_to_save[self.fn] = self.markers

        # plt.tight_layout()

    def load_filenames(self):
        ls = os.listdir(self.dirname)
#        print(ls)
        self.filenames = [os.path.join(self.dirname, f) for f in ls if f.lower().endswith(
            ('.png', '.jpg', '.bmp', '.jpg', '.tif', '.tiff'))]
        self.filenames.sort()

    def load_image(self):
        print(self.curr_frame_index)
#        print(len(self.filenames))
        # self.image = plt.imread(self.filenames[self.curr_frame_index])
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plotthe image
        self.im = self.image
        self.figure.axes[0].get_images()[0].set_data(self.im)
        # if no markers are set, use previous markers as default
        if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
            self.markers[:, self.curr_frame_index] = self.markers[:,
                                                                  self.curr_frame_index - 1]
        # then plot the markers
        for i in range(self.num_markers):
            self.figure.axes[0].lines[i].set_xdata(
                self.markers[i, self.curr_frame_index, 0:1])
            self.figure.axes[0].lines[i].set_ydata(
                self.markers[i, self.curr_frame_index, 1:2])
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        plt.draw()

    def change_frame(self, new_frame):
        print('change_frame {} {}'.format(new_frame, int(new_frame)))
        self.curr_frame_index = int(new_frame)-1
        self.load_image()
        # self.xlim = self.figure.axes[0].get_xlim()
        # self.ylim = self.figure.axes[0].get_ylim()
        self.show_image()
        # self.figure.axes[0].set_xlim(*self.xlim)
        # self.figure.axes[0].set_ylim(*self.ylim)
        if self.data_changed:
            self.save_data()
            self.data_changed = False

    def nudge(self, direction):
        self.markers[self.curr_marker,
                     self.curr_frame_index, 0] += direction.real
        self.markers[self.curr_marker,
                     self.curr_frame_index, 1] += direction.imag
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print(self.curr_frame_index)
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)

        # marker change
        elif event.key in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
            # elif int(event.key) in self.range_markers:
            if self.curr_marker + 1 <= self.markers.shape[0]:
                # white out the old marker
                self.radiobuttons.circles[self.curr_marker].set_facecolor(
                    (1.0, 1.0, 1.0, 1.0))

            # update the marker index
            if int(event.key) > 0:
                self.curr_marker = int(event.key) - 1
            else:
                self.curr_marker = 9

            if self.curr_marker + 1 > self.markers.shape[0]:
                print(
                    f"marker = {self.curr_marker} is out of the range of {self.markers.shape[0]} defined markers.")
            else:
                # place the marker
                self.markers[self.curr_marker, self.curr_frame_index, :] = [
                    event.xdata, event.ydata]
                # black in the new marker
                self.radiobuttons.circles[self.curr_marker].set_facecolor(
                    (0.0, 0.0, 0.0, 1.0))
                self.data_changed = True
                self.show_image()

        elif event.key == " ":
            self.data_changed = True
            self.markers[self.curr_marker, self.curr_frame_index, :] = [
                event.xdata, event.ydata]
            self.show_image()

        elif event.key in ('ctrl+1', 'ctrl+2', 'ctrl+3', 'ctrl+4', 'ctrl+5', 'ctrl+6'):
            # white out the old marker
            self.radiobuttons.circles[self.curr_marker].set_facecolor(
                (1.0, 1.0, 1.0, 1.0))
            # update the marker index
            self.curr_marker = int(event.key[-1]) - 1
            # place the marker
            self.markers[self.curr_marker, self.curr_frame_index, :] = [-1, -1]
            # black in the new marker
            self.radiobuttons.circles[self.curr_marker].set_facecolor(
                (0.0, 0.0, 0.0, 1.0))
            self.data_changed = True
            self.show_image()

        elif event.key == "backspace":
            self.data_changed = True
            self.markers[self.curr_marker, self.curr_frame_index, :] = [-1, -1]
            self.show_image()

        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)

    def update_sliders(self, val):
        self.show_image()

    def on_mouse_release(self, event):
        self.markers[self.curr_marker, self.curr_frame_index, :] = [
            event.xdata, event.ydata]
        # self.change_frame(0)

    def marker_button(self, lab):
        self.curr_marker = int(lab)-1
        self.show_image()

    def save_data(self):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)


species_folders = os.listdir("./")
species_folders = [folder for folder in species_folders if os.path.isdir(folder) and not folder.startswith("_")]
for folder in species_folders:
    print(f"{folder}")
    fn = os.path.join(folder, 'data.npy')
    if os.path.exists(fn):
        data = np.load(fn)
        if data.shape[0] > 6:
            new_data = data[:6]
            np.save(fn, new_data)
    tracker = tracker_window(
        dirname=folder,
        num_markers=6,
        fn='data.npy')
    plt.show()
