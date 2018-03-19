from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
from keras.models import load_model

class Paint(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.setUI()
        self.brush_size = 4
        self.brush_color = "black"

    def setUI(self):
        self.parent.title("MNIST")
        self.pack(fill=BOTH, expand=1)

        self.canv = Canvas(self, width=440, height=430, bg="white")
        self.canv.place(x=10, y=10)
        self.canv.bind("<B1-Motion>", self.draw)

        self.image = Image.new("RGB", (440, 430), "white")
        self.dr = ImageDraw.Draw(self.image)

        btn_prediction = Button(self, text="Clear", width=10, height=2, bg="white", fg="black", command=lambda: self.clear())
        btn_prediction.place(x=490, y=300)

        btn_prediction = Button(self, text="Predict", width=10, height=2, bg="white", fg="black", command=lambda: self.predict())
        btn_prediction.place(x=490, y=200)

        self.lbl_answer = Label(self, width=10, height=2, bg="white", fg="black")
        self.lbl_answer.place(x=490, y=150)

    def clear(self):
        self.canv.delete("all")
        self.lbl_answer["text"] = ''

    def draw(self, event):
         self.dr.ellipse((event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size),
                              fill="black", outline="black")

         self.canv.create_oval(event.x - self.brush_size,
                            event.y - self.brush_size,
                            event.x + self.brush_size,
                            event.y + self.brush_size,
                            fill="black", outline="black")

    def resize_image(self):
        file_name = "test.png"
        basewidth = 28
        img = Image.open(file_name)
        hsize = 28
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save('s.png')

    def save_image(self, img):
        del self.dr
        img.save('test.png', 'PNG')
        self.resize_image()

    def imageprepare(self):
        im = Image.open('s.png')
        im_grey = im.convert('L')
        im_array = np.array(im_grey)
        im_array = np.reshape(im_array, (1, 784)).astype('float32')
        x = 255 - im_array
        x /= 255
        return x


    def predict(self):
        self.save_image(self.image)
        model = load_model('mymodel')
        x = self.imageprepare()
        prediction = model.predict(x)
        self.lbl_answer["text"] = np.argmax(prediction, axis=1)
        self.dr = ImageDraw.Draw(self.image)


def main():
    root = Tk()
    root.geometry("600x450+300+300")
    app = Paint(root)
    root.mainloop()

if __name__ == "__main__":
    main()