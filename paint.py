from PIL import ImageTk, ImageDraw, ImageFont
import PIL
from tkinter import *
import numpy as np
import sketch_to_code

width = 500
height = 500
center = height // 2
white = (255, 255, 255)
green = (0, 128, 0)
root = Tk()
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()
my_text = ""


def trainingImage():
    random = np.random.randint(low=1, high=1000, size=1)
    filename = "C://Users/hp/Desktop/ai_python/training_cnn/images/sketch" + str(random[0]) + ".png"
    image1.save(filename)


def trainingRadio():
    random = np.random.randint(low=1, high=1000, size=1)
    filename = "C://Users/hp/Desktop/ai_python/training_cnn/radioButton/sketch" + str(random[0]) + ".png"
    image1.save(filename)


def trainingCheckBox():
    random = np.random.randint(low=1, high=1000, size=1)
    filename = "C://Users/hp/Desktop/ai_python/training_cnn/checkBox/sketch" + str(random[0]) + ".png"
    image1.save(filename)


def testingImage():
    random = np.random.randint(low=1, high=1000, size=1)
    filename = "C://Users/hp/Desktop/ai_python/testing_cnn/images/sketch" + str(random[0]) + ".png"
    image1.save(filename)


def testingRadio():
    random = np.random.randint(low=1, high=1000, size=1)
    filename = "C://Users/hp/Desktop/ai_python/testing_cnn/radioButton/sketch" + str(random[0]) + ".png"
    image1.save(filename)


def testingCheckBox():
    random = np.random.randint(low=1, high=1000, size=1)
    filename = "C://Users/hp/Desktop/ai_python/testing_cnn/checkBox/sketch" + str(random[0]) + ".png"
    image1.save(filename)


def predict():
    filename = "sketch1.png"
    image1.save(filename)
    result = sketch_to_code.runModel()
    giveResult(result[0])


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)


def giveResult(value):
    my_text = getCode(value);
    print(my_text)
    cv.create_text(100, 40, fill="darkblue", font="Times 14 italic bold",
                   text=my_text)
    with open("Output.txt", "w") as text_file:
        text_file.write(getCode(value))


def exit():
    root.destroy()


def clean():
    my_text = "";
    cv.delete("all")
    global image1
    image1 = PIL.Image.new("RGB", (width, height), white)
    global draw
    draw = ImageDraw.Draw(image1)


def getCode(value):
    switcher = {
        0: "<check-box checked></check-box>",
        1: "<img height=100 width=100 src=''>",
        2: "<radio-button ></radio-button>"
    }

    return switcher.get(value, "Invalid data")


image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)
cv.create_text(100, 40, fill="darkblue", font="Times 14 italic bold",
               text=my_text)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# button = Button(text="training an image", command=trainingImage)
# button.pack()
# button = Button(text="testing an image", command=testingImage)
# button.pack()

# button = Button(text="training a RadioButton", command=trainingRadio)
# button.pack()
# button = Button(text="testing a RadioButton", command=testingRadio)
# button.pack()
#
# button = Button(text="training a CheckBox", command=trainingCheckBox)
# button.pack()
# button = Button(text="testing a CheckBox", command=testingCheckBox)
# button.pack()

button = Button(text="predict", command=predict)
button.pack()
button = Button(text="result", command=giveResult)
button.pack()
button = Button(text="clean", command=clean)
button.pack()
button = Button(text="exit", command=exit)
button.pack()
root.mainloop()
