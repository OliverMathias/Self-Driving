import io
import time
import datetime
import picamera
from PIL import Image
import RPi.GPIO as GPIO
import serial
import tkinter
from tkinter.ttk import *
from multiprocessing import Process

def calculate_angle(clean_line):
    #this var should be negative if the camera is turned to face the right 
    #and vice versa
    #try 7.5 as a good offset for the marks you made
    offset_angle = 7.5
    try:
        zone = int(clean_line[0])
        mid_angle = int(clean_line[1])
        if zone > 125:
        #then the wheel is turning left
        #this random ass number is the fixed float that most closely gets us an accurate degree
          temp_angle = (((zone-255) * 255) + (mid_angle - 255))/9.28888888889
          return temp_angle + offset_angle

        else:
          temp_angle = (((zone * 255) + mid_angle))/9.28888888889
          return temp_angle + offset_angle
    except:
        pass

def get_angle_and_speed():
    global ser
    # send an R to the arduino
    test = "R"
    ser.write(test.encode())
    #read data from arduino
    line = (ser.readline())
    clean_line = line.split()
    try:
        raw_speed = int(clean_line[2])
    except:
        raw_speed = 0
    return calculate_angle(clean_line[:2]), (raw_speed*0.8125)

def write_to_file(image_name, angle, speed):
    text_to_write = image_name + "," + str(angle) + "," + str(speed) + "\n"
    with open("/home/pi/Desktop/data.txt", "a") as myfile:
        try:
            myfile.write(text_to_write)
        except:
            myfile.write("na")
        myfile.flush()
        myfile.close()


def save_image_to_folder(image, filename):
    image.save("/home/pi/Desktop/data/" + filename + ".jpeg")

def outputs():
    stream = io.BytesIO()
    for i in range(10):
        # This returns the stream for the camera to capture to
        yield stream
        #gets start of stream
        stream.seek(0)
        #opens stream as PIL image
        img = Image.open(stream)
        
        angle, speed = get_angle_and_speed()
        
        #inits filename for 2 methods to use
        #prepend the img number from global
        global last_image_number_taken;
        last_image_number_taken += 1

        filename = str(last_image_number_taken) + "-" + str(datetime.datetime.now())

        processes = []

        processes.append(Process(target=save_image_to_folder, args=(img, filename)))
        processes.append(Process(target=write_to_file, args=(filename,angle,speed)))

        
        for process in processes:
            process.start()

        # Finally, reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
        
        time.sleep(0.04)

        #finish saving all pics and data before next batch
        if ((i % 2 == 0) and (i > 0)):
            for process in processes:
                process.join()

def start():
    global path
    global capture_data
    global last_image_number_taken
    
    capture_data = True
    print("Recording Started @: ", str(datetime.datetime.now()))

    with open(path,'r') as f:
        last_image_number_taken = int(f.readline())
        print("Last Image Number: ",last_image_number_taken)
        f.close()


def stop():
    global path
    global capture_data
    global last_image_number_taken

    capture_data = False
    print("Recording Stopped @: ", str(datetime.datetime.now()))

    with open(path,'w+') as f:
        print("Writing Last Image Number: ", last_image_number_taken)
        f.write('%d' % int(last_image_number_taken))
        f.close()


if __name__ == '__main__':

    num_sequences = 0
    time_spent = 0

    #switch gpio settings
    #GPIO Basic initialization
    #Set warnings to false
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    #Initialize GPIO Pin 4 as an input pin
    #Also, PUD down makes the off value 0
    GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    root = tkinter.Tk()
    root.geometry('250x500')

    style = Style()
    style.configure('TButton', font =
                   ('calibri', 35, 'bold'),
                        borderwidth = '4')

    # start button
    btn1 = tkinter.Button(root, height=10,width=25, bg="green", font=('Helvetica','12'), text = 'Start', command = start)
    btn1.grid(row = 0, column = 1, padx = 1)

    # stop button
    btn2 = tkinter.Button(root, height=10,width=25, bg="red", font=('Helvetica','12'), text = 'Stop', command = stop)
    btn2.grid(row = 1, column = 1, pady = 10, padx = 1)

    #load in last image number from file storage
    #set path here and have getters and setters in start and stop functions
    path = '/home/pi/Desktop/last_image_number.txt'
    last_image_number_taken = 0

    with open(path,'r') as f:
        last_image_number_taken = f.readline()
        print("Last Image Number: ",last_image_number_taken)
        f.close()

    #initialize loop control variables
    capture_data = False

    #serial connection with arduino
    ser = serial.Serial(port='/dev/ttyACM0',baudrate = 9600,parity=serial.PARITY_NONE,
           stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS,timeout=1)

    with picamera.PiCamera() as camera:

        #init camera and let shutter adjust
        camera.resolution = (960, 520)
        camera.framerate = 80
        camera.start_preview(fullscreen=False, window=(350,300,400,180))
        time.sleep(2)

        while True:

            if GPIO.input(10) == 1:
                print("Turning...")
                time.sleep(10)
                print("Resume Recording @: ", str(datetime.datetime.now()))

            else:
                #updates screen
                root.update()

                if capture_data:
                    start = time.time()
                    camera.capture_sequence(outputs(), 'jpeg', use_video_port=True)
                    finish = time.time()
                    num_sequences += 1
                    time_spent += (finish-start)
                    print("Average Capture Time: ",time_spent/num_sequences)
