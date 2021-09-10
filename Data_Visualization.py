import numpy as np
from tkinter import Tk, IntVar, StringVar, Frame, LabelFrame, Label, Listbox, Button, Entry, ACTIVE, DISABLED, \
    TOP, LEFT, BOTTOM, BOTH, CENTER
from tkinter.messagebox import showerror
from PIL import Image, ImageTk
from Data_PreProcessing import load_captions_as_dataframe
import os

os.chdir(os.path.dirname(__file__))


def visualize_data(data=load_captions_as_dataframe()):
    images_path = 'Flickr30k/images/'
    unique_images = np.unique(data.image_name.values)

    root = Tk()
    root.geometry('1400x720+50+50')
    root.title('Data Visualization')
    root.resizable(0, 0)

    count = IntVar()
    img_name = StringVar()

    count.set(0)
    image = ImageTk.PhotoImage(Image.open(images_path + unique_images[count.get()]).resize((500, 500)))
    img_name.set(unique_images[count.get()])

    def modify_image():
        image_lbl.image = ImageTk.PhotoImage(Image.open(images_path + unique_images[count.get()]).resize((500, 500)))
        img_name.set(unique_images[count.get()])
        image_lbl.configure(image=image_lbl.image)

    def listbox_insert(listbox):
        if listbox.size() != 0:
            listbox.delete(0, listbox.size() - 1)
        temp = list(data["caption"].loc[data["image_name"] == unique_images[count.get()]].values)
        pos = 0
        for caption in temp:
            listbox.insert(pos, caption)
            pos += 1

    def turn_right(listbox):
        left_btn.configure(state=ACTIVE)
        count.set(count.get() + 1)
        if count.get() == 31782:
            right_btn.configure(state=DISABLED)
        modify_image()
        listbox_insert(listbox)

    def turn_left(listbox):
        if count.get() == 0:
            left_btn.config(state=DISABLED)
        else:
            right_btn.configure(state=ACTIVE)
            count.set(count.get() - 1)
            if count.get() == 0:
                left_btn.configure(state=DISABLED)
            modify_image()
            listbox_insert(listbox)

    def search(file_name, listbox):
        if file_name == '':
            showerror(title='Warning', message='Input the image name first!')
        elif file_name not in unique_images:
            showerror(title='Error', message='There is no image with the name you have entered!')
        else:
            count.set(np.where(unique_images == file_name)[0][0])
            if count.get() == 0:
                left_btn.configure(state=DISABLED)
            elif count.get() == 31782:
                right_btn.configure(state=DISABLED)
            else:
                left_btn.configure(state=ACTIVE)
                right_btn.configure(state=ACTIVE)
            modify_image()
            listbox_insert(listbox)

    header = Label(root, text=f'Flickr30K Dataset - {len(unique_images)} images', font='arial 20 bold', justify=CENTER,
                   bg='black', fg='white', height=2)
    middle = Frame(root)
    tail = Frame(root, bg='black')

    image_name = Label(tail, textvariable=img_name, font='arial 20 bold', justify=CENTER, bg='black', fg='white',
                       height=2)
    image_name.pack(fill=BOTH)

    imgFrame = Frame(middle)
    image_lbl = Label(imgFrame, image=image)
    captions_and_tools = Frame(middle)

    captions_frame = LabelFrame(captions_and_tools, text='Captions :')
    captions_list = Listbox(captions_frame, font='arial 15 bold', selectbackground='gray', selectforeground='black',
                            borderwidth=3, selectborderwidth=4, activestyle='none', height=5)
    listbox_insert(captions_list)
    captions_list.pack(fill=BOTH)
    captions_frame.pack(fill=BOTH)

    search_frame = LabelFrame(captions_and_tools, text='Search By Image Name :')

    box = Entry(search_frame, font='arial 15 bold', selectbackground='gray', selectforeground='black')
    search_btn = Button(search_frame, text='Search', font='arial 15 bold', bg='black', fg='white',
                        command=lambda: search(box.get(), captions_list))
    box.pack(fill=BOTH, side=LEFT, padx=(10, 20), pady=20)
    search_btn.pack(fill=BOTH, side=LEFT, pady=20)
    search_frame.pack(fill=BOTH, pady=(20, 90))

    tools = LabelFrame(captions_and_tools, text='Navigate Through the data :')

    right_icon = ImageTk.PhotoImage(Image.open('icons/icons8-left-curving-right-96.png').resize((60, 60)))
    left_icon = ImageTk.PhotoImage(Image.open('icons/icons8-right-curving-left-96.png').resize((60, 60)))
    right_btn = Button(tools, image=right_icon, bd=0, command=lambda: turn_right(captions_list))
    left_btn = Button(tools, image=left_icon, bd=0, command=lambda: turn_left(captions_list), state=DISABLED)
    left_btn.pack(side=LEFT, padx=40, pady=15)
    right_btn.pack(side=LEFT, padx=40, pady=15)
    tools.pack(fill=BOTH, side=BOTTOM)

    imgFrame.pack(fill=BOTH, side=LEFT, pady=(35, 35))
    image_lbl.pack()
    captions_and_tools.pack(fill=BOTH, pady=30)

    header.pack(side=TOP, fill=BOTH)
    middle.pack(side=TOP, fill=BOTH)
    tail.pack(side=BOTTOM, fill=BOTH)

    root.bind('<Escape>', lambda *args: root.destroy())
    root.bind('<Shift-Left>', lambda *args: turn_left(captions_list))
    root.bind('<Shift-Right>', lambda *args: turn_right(captions_list))
    root.mainloop()


visualize_data()
