# raspberry_pi_controller.py

import RPi.GPIO as GPIO
import time
# define pins numbering scheme (physical (board) or gpio (bcm))
GPIO.setmode(GPIO.BOARD)
# (or) GPIO.setmode(GPIO.BCM)

def main():
    print('starting GPIO 11 and 13.')
    # define output pins
    GPIO.setup(11, GPIO.OUT)
    GPIO.setup(13, GPIO.OUT)

    # send signal throght output pin
    GPIO.output(11, True) # or 1 instead of True
    GPIO.output(13, True) # or 1 instead of True
    #GPIO.output(11, False) # or 0 instead of False

    i=0
    seconds = 60
    while i<seconds:
        i+=1
        print(f'sleep second {i}/{seconds}')
        time.sleep(1)

    # release definition from pins
    GPIO.cleanup()

if __name__ == '__main__':
    main()
