import pygame
import numpy as np
from game import AutoDriveGame, Car

if __name__ == "__main__":
    pygame.init()
    display = pygame.display.set_mode((640, 640))

    car = Car(0, (50, 50), (17, 29), (255, 125, 0), 1, 5, 30, 0.01)
    map_ = np.zeros((640, 640, 3))
    map_[205:209] = 255
    map_[205:209, 240: 320] = 0
    map_[295:299] = 255
    map_[295:299, 140: 210] = 0
    map_[395:399] = 255
    map_[395:399, 340: 390] = 0

    car_game = AutoDriveGame(map_, car, (500, 500), max_steps=1024)

    while True:
        action = [0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = -1
                elif event.key == pygame.K_LEFT:
                    action[1] = 1
                elif event.key == pygame.K_RIGHT:
                    action[1] = -1

        else:
            if not car_game.step(action):
                print("score:", car_game.get_score())
                car_game.reset()

            surf = pygame.surfarray.make_surface(car_game.view)
            display.blit(surf, (0, 0))
            pygame.display.update()
            continue
        break
    pygame.quit()





