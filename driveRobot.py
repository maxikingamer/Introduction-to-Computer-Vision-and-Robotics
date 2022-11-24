import curses
import ev3_dc as ev3


def main(stdscr) -> None:
    '''
    controls terminal and keyboard events
    '''
    def react():
        '''
        reacts on keyboard arrow key events by modifying speed and turn
        '''
        nonlocal speed, turn
        if c == curses.KEY_LEFT:
            turn += 5
            turn = min(turn, 200)
        elif c == curses.KEY_RIGHT:
            turn -= 5
            turn = max(turn, -200)
        elif c == curses.KEY_UP:
            speed += 5
            speed = min(speed, 100)
        elif c == curses.KEY_DOWN:
            speed -= 5
            speed = max(speed, -100)

    # initialize terminal

    stdscr.clear()
    stdscr.refresh()
    stdscr.addstr(0, 0, 'use Arrows to navigate your vehicle')
    stdscr.addstr(1, 0, 'pause your vehicle with key <p>')
    stdscr.addstr(2, 0, 'terminate with key <q>')

    # control vehicle movement and visualize it

    speed = 0
    turn = 0
    with ev3.TwoWheelVehicle(
        0.03575,  # radius_wheel
        0.18641,  # tread
        protocol=ev3.USB,
    ) as my_vehicle:
        while True:
            c = stdscr.getch()  # catch keyboard event
            if c in (
                curses.KEY_RIGHT,
                curses.KEY_LEFT,
                curses.KEY_UP,
                curses.KEY_DOWN
            ):
                react()
                my_vehicle.move(speed, -turn)  # modify movement
                stdscr.addstr(
                    4,
                    0,
                    f'speed: {speed:4d}, turn: {turn:4d}          '
                )
            elif c == ord('p'):
                speed = 0
                turn = 0
                my_vehicle.stop()  # stop movement
                pos = my_vehicle.position
                stdscr.addstr(
                    4,
                    0,
                    f'x: {pos.x:5.2f} m, y: {pos.y:5.2f} m, o: {pos.o:4.0f} °'
                )
            elif c in (ord('q'), 27):
                my_vehicle.stop()  # finally stop movement
                break

curses.wrapper(main)