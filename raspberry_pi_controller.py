# raspberry_pi_controller.py

import RPi.GPIO as GPIO
import time
# define pins/pwms numbering scheme (physical (board) or gpio (bcm))
GPIO.setmode(GPIO.BOARD)
# (or) GPIO.setmode(GPIO.BCM)

def main():
    print('starting GPIO 11 and 13.')
    # define output pwms
    pwm_nums = np.array([11,12,13,15])
    for num in pwm_nums:
        GPIO.setup(num, GPIO.OUT)

    # send signal throght output pin
    #GPIO.output(11, True) # or 1 instead of True
    #GPIO.output(11, False) # or 0 instead of False

    # controll the signal voltage (dim)
    pwms = set()
    for num in pwm_nums:
        # PWM(pin, freq) pin_num/GPIO_num, requency
        pwm = GPIO.PWM(num, 1000)
        # start(dutyCycle), dutyCycle = [0,100] continuous range
        pwm.start(100)
        #  ChangeFrequency(freq) freq = new frequency in Hertz
        #pwm.ChangeFrequency(freq)
        # ChangeDutyCycle(dutyCycle) dutyCycle = [0,100] continuous range
        #pwm.ChangeDutyCycle(100)
        pwms.add(pwm)

    # send UP signal for 60 seconds
    i=0
    seconds = 60
    while i<seconds:
        i+=1
        print(f'sleep second {i}/{seconds}')
        time.sleep(1)


    # release definition from pwms
    for pwm in pwms:
        pwm.stop()
    GPIO.cleanup()

if __name__ == '__main__':
    main()
