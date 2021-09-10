from tkinter import Tk, StringVar, Frame, LabelFrame, Label, Listbox, Button, ACTIVE, DISABLED, LEFT, BOTH, CENTER, \
    filedialog, BOTTOM
from tkinter.messagebox import showerror
from PIL import ImageTk
from Evaluating import *

os.chdir(os.path.dirname(__file__))

root = Tk()
root.geometry('1400x720+50+50')
root.title('Image Caption Generator')
root.resizable(0, 0)

training_data = load_captions_as_dataframe()
testing_data = load_dataframe_of_testImages_and_captions_stage2()
images_path = 'Flickr30k/images/'
unique_training_images = np.unique(training_data.image_name.values)
unique_testing_images = np.unique(testing_data.image_name.values)
img_name = StringVar()
uploaded_image_path = StringVar()
uploaded_image_path.set('')

img_name.set(np.random.choice(unique_testing_images, 1)[0])
image = ImageTk.PhotoImage(Image.open(images_path + img_name.get()).resize((500, 500)))


def random_trainImage():
    cleaning_captions_slots()
    uploaded_image_path.set('')
    random_image = np.random.choice(unique_training_images, 1)[0]
    img_name.set(random_image)
    modify_image()
    modify_captions(captions_list)
    disable_attention_buttons()


def random_testImage():
    cleaning_captions_slots()
    uploaded_image_path.set('')
    random_image = np.random.choice(unique_testing_images, 1)[0]
    img_name.set(random_image)
    modify_image()
    modify_captions(captions_list)
    disable_attention_buttons()


def upload_image():
    cleaning_captions_slots()
    filename = filedialog.askopenfilename(initialdir="./", title="Select An Image", filetypes=(
        ("jpeg files", "*.jpg"), ("gif files", "*.gif*"), ("png files", "*.png")))
    uploaded_image_path.set(filename)
    img_name.set(filename.split('/')[-1])
    image_lbl.image = ImageTk.PhotoImage(Image.open(filename).resize((500, 500)))
    image_lbl.configure(image=image_lbl.image)
    if captions_list.size() != 0:
        captions_list.delete(0, captions_list.size() - 1)
    disable_attention_buttons()


def modify_image():
    image_lbl.image = ImageTk.PhotoImage(Image.open(images_path + img_name.get()).resize((500, 500)))
    image_lbl.configure(image=image_lbl.image)


def modify_captions(listbox):
    if listbox.size() != 0:
        listbox.delete(0, listbox.size() - 1)
    temp = list(training_data["caption"].loc[training_data["image_name"] == img_name.get()])
    pos = 0
    for caption in temp:
        listbox.insert(pos, caption)
        pos += 1


def modify_greedy():
    if greedy_Caption.size() != 0:
        greedy_Caption.delete(0, greedy_Caption.size() - 1)

    if uploaded_image_path.get() == '':
        caption, attention_plot = evaluate_greedy(images_path + img_name.get())
    else:
        caption, attention_plot = evaluate_greedy(uploaded_image_path.get())

    cleaned_caption = [word for word in caption if not word in ['<start>', '<end>', '<unk>']]
    greedy_Caption.insert(0, cleaned_caption)
    greedy_visualize_attention.configure(state=ACTIVE, activeforeground='white', activebackground='black')
    global greedy_caption, greedy_attention
    greedy_caption = caption
    greedy_attention = attention_plot


def modify_beam():
    if beam_Caption.size() != 0:
        beam_Caption.delete(0, beam_Caption.size() - 1)

    if uploaded_image_path.get() == '':
        caption, attention_plot = evaluate_beam(images_path + img_name.get())
    else:
        caption, attention_plot = evaluate_beam(uploaded_image_path.get())

    cleaned_caption = [word for word in caption if not word in ['<start>', '<end>', '<unk>']]
    beam_Caption.insert(0, cleaned_caption)
    beam_visualize_attention.configure(state=ACTIVE, activeforeground='white', activebackground='black')
    global beam_caption, beam_attention
    beam_caption = caption
    beam_attention = attention_plot


def modify_topk_with_temperature():
    if topk_Caption.size() != 0:
        topk_Caption.delete(0, topk_Caption.size() - 1)

    if uploaded_image_path.get() == '':
        caption, attention_plot = evaluate_topK(images_path + img_name.get())
    else:
        caption, attention_plot = evaluate_topK(uploaded_image_path.get())

    cleaned_caption = [word for word in caption if not word in ['<start>', '<end>', '<unk>']]
    topk_Caption.insert(0, cleaned_caption)
    topk_visualize_attention.configure(state=ACTIVE, activeforeground='white', activebackground='black')
    global topk_caption, topk_attention
    topk_caption = caption
    topk_attention = attention_plot


def modify_nucleus():
    if nucleus_Caption.size() != 0:
        nucleus_Caption.delete(0, nucleus_Caption.size() - 1)

    if uploaded_image_path.get() == '':
        caption, attention_plot = evaluate_nucleus(images_path + img_name.get())
    else:
        caption, attention_plot = evaluate_nucleus(uploaded_image_path.get())

    cleaned_caption = [word for word in caption if not word in ['<start>', '<end>', '<unk>']]
    nucleus_Caption.insert(0, cleaned_caption)
    nucleus_visualize_attention.configure(state=ACTIVE, activeforeground='white', activebackground='black')
    global nucleus_caption, nucleus_attention
    nucleus_caption = caption
    nucleus_attention = attention_plot


def greedy_visualize():
    cleaned_caption = [word for word in greedy_caption if word not in ['<start>', '<end>', '<unk>']]
    if len(cleaned_caption) <= 16:
        if uploaded_image_path.get() == '':
            plot_attention(images_path + img_name.get(), greedy_caption, greedy_attention)
        else:
            plot_attention(uploaded_image_path.get(), greedy_caption, greedy_attention)
    else:
        showerror(title='Warning',
                  message='Attention Visualization was configured to work only with captions of length equal or lower '
                          'than 16')


def beam_visualize():
    cleaned_caption = [word for word in beam_caption if word not in ['<start>', '<end>', '<unk>']]
    if len(cleaned_caption) <= 16:
        if uploaded_image_path.get() == '':
            plot_attention(images_path + img_name.get(), beam_caption, beam_attention)
        else:
            plot_attention(uploaded_image_path.get(), beam_caption, beam_attention)
    else:
        showerror(title='Warning',
                  message='Attention Visualization was configured to work only with captions of length equal or lower '
                          'than 16')


def topk_visualize():
    cleaned_caption = [word for word in topk_caption if word not in ['<start>', '<end>', '<unk>']]
    if len(cleaned_caption) <= 16:
        if uploaded_image_path.get() == '':
            plot_attention(images_path + img_name.get(), topk_caption, topk_attention)
        else:
            plot_attention(uploaded_image_path.get(), topk_caption, topk_attention)
    else:
        showerror(title='Warning',
                  message='Attention Visualization was configured to work only with captions of length equal or lower '
                          'than 16')


def nucleus_visualize():
    cleaned_caption = [word for word in nucleus_caption if word not in ['<start>', '<end>', '<unk>']]
    if len(cleaned_caption) <= 16:
        if uploaded_image_path.get() == '':
            plot_attention(images_path + img_name.get(), nucleus_caption, nucleus_attention)
        else:
            plot_attention(uploaded_image_path.get(), nucleus_caption, nucleus_attention)
    else:
        showerror(title='Warning',
                  message='Attention Visualization was configured to work only with captions of length equal or lower '
                          'than 16')


# (1) Header
header = Label(root, text='Image Caption Generator', font='arial 20 bold', justify=CENTER, bg='black', fg='white',
               height=2)

# (2) Images-Captions-Decoding_Strategies LabelFrame:
images_captions_decodingStrategies = LabelFrame(root, text='Images and Captions :')

# (2-1) Image-AND-its-name Frame
image_frame = Frame(images_captions_decodingStrategies)
image_lbl = Label(image_frame, image=image)
image_name = Label(image_frame, textvariable=img_name, font='arial 20 bold', justify=CENTER, bg='black', fg='white',
                   height=2)
image_lbl.pack()
image_name.pack(fill=BOTH)
image_frame.pack(side=LEFT)

# (2-2) Captions-Decoding_Strategies Frame
captions_decodingStrategies = Frame(images_captions_decodingStrategies)

# (2-2-1) Captions Listbox
captions_frame = LabelFrame(captions_decodingStrategies, text='Captions: ')
captions_list = Listbox(captions_frame, font='arial 15 bold', selectbackground='gray',
                        selectforeground='black', borderwidth=3, selectborderwidth=4, activestyle='none', height=5)
modify_captions(captions_list)
captions_list.pack(fill=BOTH)
captions_frame.pack(fill=BOTH)

# (2-2-2) Greedy Search:
greedy_Search = LabelFrame(captions_decodingStrategies, text='Greedy Search: ')

# Buttons:
greedy_Buttons = Frame(greedy_Search)
greedy_generate_caption = Button(greedy_Buttons, text='Greedy Search', bg='black', fg='white', font='arial 10 bold',
                                 command=modify_greedy)
greedy_visualize_attention = Button(greedy_Buttons, text='Visualize Attention', bg='black', fg='white',
                                    font='arial 10 bold', command=greedy_visualize)
greedy_generate_caption.pack(fill=BOTH, padx=(10, 0), pady=(0, 2))
greedy_visualize_attention.pack(fill=BOTH, padx=(10, 0))
greedy_Buttons.pack(side=LEFT, fill=BOTH)

# Generated Caption:
greedy_Caption = Listbox(greedy_Search, font='arial 15 bold', selectbackground='gray',
                         selectforeground='black', borderwidth=3, selectborderwidth=4, activestyle='none', height=1)
greedy_Caption.pack(fill=BOTH, padx=(15, 0))

greedy_Search.pack(fill=BOTH, pady=(15, 0))

# (2-2-3) Beam Search:
beam_Search = LabelFrame(captions_decodingStrategies, text='Beam Search (B=10): ')

# Buttons:
beam_Buttons = Frame(beam_Search)
beam_generate_caption = Button(beam_Buttons, text='Beam Search', bg='black', fg='white', font='arial 10 bold',
                               command=modify_beam)
beam_visualize_attention = Button(beam_Buttons, text='Visualize Attention', bg='black', fg='white',
                                  font='arial 10 bold', command=beam_visualize)
beam_generate_caption.pack(fill=BOTH, padx=(10, 0), pady=(0, 2))
beam_visualize_attention.pack(fill=BOTH, padx=(10, 0))
beam_Buttons.pack(side=LEFT, fill=BOTH)

# Generated Caption:
beam_Caption = Listbox(beam_Search, font='arial 15 bold', selectbackground='gray',
                       selectforeground='black', borderwidth=3, selectborderwidth=4, activestyle='none', height=1)
beam_Caption.pack(fill=BOTH, padx=(15, 0))

beam_Search.pack(fill=BOTH, pady=(20, 0))

# (2-2-4) Topk Sampling with Temperature:
topk_sampling = LabelFrame(captions_decodingStrategies, text='TopK Sampling with Temperature (K=25, T=0.8): ')

# Buttons:
topk_Buttons = Frame(topk_sampling)
topk_generate_caption = Button(topk_Buttons, text='TopK Sampling', bg='black', fg='white', font='arial 10 bold',
                               command=modify_topk_with_temperature)
topk_visualize_attention = Button(topk_Buttons, text='Visualize Attention', bg='black', fg='white',
                                  font='arial 10 bold', command=topk_visualize)
topk_generate_caption.pack(fill=BOTH, padx=(10, 0), pady=(0, 2))
topk_visualize_attention.pack(fill=BOTH, padx=(10, 0))
topk_Buttons.pack(side=LEFT, fill=BOTH)

# Generated Caption:
topk_Caption = Listbox(topk_sampling, font='arial 15 bold', selectbackground='gray',
                       selectforeground='black', borderwidth=3, selectborderwidth=4, activestyle='none', height=1)
topk_Caption.pack(fill=BOTH, padx=(15, 0))

topk_sampling.pack(fill=BOTH, pady=(20, 0))

# (2-2-5) Nucleus Sampling:
nucleus_sampling = LabelFrame(captions_decodingStrategies, text='Nucleus Sampling (P=0.9): ')

# Buttons:
nucleus_Buttons = Frame(nucleus_sampling)
nucleus_generate_caption = Button(nucleus_Buttons, text='Nucleus Sampling', bg='black', fg='white',
                                  font='arial 10 bold', command=modify_nucleus)
nucleus_visualize_attention = Button(nucleus_Buttons, text='Visualize Attention', bg='black', fg='white',
                                     font='arial 10 bold', command=nucleus_visualize)
nucleus_generate_caption.pack(fill=BOTH, padx=(10, 0), pady=(0, 2))
nucleus_visualize_attention.pack(fill=BOTH, padx=(10, 0))
nucleus_Buttons.pack(side=LEFT, fill=BOTH)

# Generated Caption:
nucleus_Caption = Listbox(nucleus_sampling, font='arial 15 bold', selectbackground='gray',
                          selectforeground='black', borderwidth=3, selectborderwidth=4, activestyle='none', height=1)
nucleus_Caption.pack(fill=BOTH, padx=(15, 0))

nucleus_sampling.pack(fill=BOTH, pady=(20, 0))

captions_decodingStrategies.pack(fill=BOTH)

# (3) Tail
tail = Label(root, bg='black')
random_trainImage_button = Button(tail, text='Random Train Image', bg='black', fg='white', font='arial 13',
                                  command=random_trainImage)
random_testImage_button = Button(tail, text='Random Test Image', bg='black', fg='white', font='arial 13',
                                 command=random_testImage)
upload_image_button = Button(tail, text='Upload Image', bg='black', fg='white', font='arial 13',
                             command=upload_image)
random_trainImage_button.pack(side=LEFT, padx=(2, 0))
random_testImage_button.pack(side=LEFT, padx=(28, 0))
upload_image_button.pack(side=LEFT, padx=(28, 0))

header.pack(fill=BOTH)
images_captions_decodingStrategies.pack(fill=BOTH)
tail.pack(fill=BOTH, side=BOTTOM)


def disable_attention_buttons():
    greedy_visualize_attention.configure(state=DISABLED)
    beam_visualize_attention.configure(state=DISABLED)
    topk_visualize_attention.configure(state=DISABLED)
    nucleus_visualize_attention.configure(state=DISABLED)


def cleaning_captions_slots():
    if greedy_Caption.size() != 0:
        greedy_Caption.delete(0, greedy_Caption.size() - 1)

    if beam_Caption.size() != 0:
        beam_Caption.delete(0, beam_Caption.size() - 1)

    if topk_Caption.size() != 0:
        topk_Caption.delete(0, topk_Caption.size() - 1)

    if nucleus_Caption.size() != 0:
        nucleus_Caption.delete(0, nucleus_Caption.size() - 1)


disable_attention_buttons()

root.bind('<Escape>', lambda *args: root.destroy())
root.bind('<Shift-Left>', lambda *args: random_trainImage())
root.bind('<Shift-Right>', lambda *args: random_testImage())
root.bind('<Shift-U>', lambda *args: upload_image())

root.bind('<Shift-G>', lambda *args: modify_greedy())
root.bind('<Shift-B>', lambda *args: modify_beam())
root.bind('<Shift-T>', lambda *args: modify_topk_with_temperature())
root.bind('<Shift-N>', lambda *args: modify_nucleus())

root.bind('<Control-Shift-G>', lambda *args: greedy_visualize())
root.bind('<Control-Shift-B>', lambda *args: beam_visualize())
root.bind('<Control-Shift-T>', lambda *args: topk_visualize())
root.bind('<Control-Shift-N>', lambda *args: nucleus_visualize())

root.mainloop()
