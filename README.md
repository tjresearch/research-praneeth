# Automating Scorekeeping and Sports Analysis of a Basketball Game using Computer Vision
## Overview
Given the input of a video of a NBA basketball game, the project will autonomously detect the ball to calculate score and other statistics in basketball like assists and rebounds.  In the NBA today, a scorekeeper is needed at every NBA game, and humans sometimes make mistakes.  Also, some statistics like assists are up to the scorekeepers viewpoint.  They can differ around the league.  An automated process of this stats collection will help all players be on the same level of play.  The project should generate a live stat sheet while the game is being processed as the final result.  Currently, the output will be a scoreboard that is boxed.  I am working on tracking the ball.
## Requirements
- Python 3.7
- OpenCV 4.1.2.30 or later
- Numpy
- Imutils
## Installation
1. Download the video_computation.py file
2. In line 76, replace the video name to the filename of your basketball game video
## Run Instructions
If you have a valid basketball game video, and you include the filename in video_computation.py, you can run the program.
## Sample Output
The sample output will be a stat sheet along with the video of the basketball game with the ball tracked.  Right now, it is a video of the masked video to identify the ball.
