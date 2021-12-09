# raspberry_pi_controller.py

import RPi.GPIO as GPIO
import time
# define pins numbering scheme (physical (board) or gpio (bcm))
GPIO.setmode(GPIO.BOARD)
# (or) GPIO.setmode(GPIO.BCM)

def main():
    # define output pins
    GPIO.setup(11, GPIO.OUT)

    # send signal throght output pin
    GPIO.output(11, True) # or 1 instead of True
    #GPIO.output(11, False) # or 0 instead of False

    time.sleep(20)
    # release definition from pins
    GPIO.cleanup()

if __name__ == '__main__':
    main()