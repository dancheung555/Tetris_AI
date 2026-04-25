# Tetris_AI
Create an AI that is really good at Tetris.

This will use both statistical and machine learning knowledge I currently possess. In many ways such as tracking what tetromino is coming up to planning where it place what piece, I will dedicate myself to work my way from typing my first line of import to the integration blend to finalize my Tetris AI.

# Background
I have been interested in Tetris since middle school, like 8th grade, and started playing Tetris time to time in high school and university. Now, currently near the end of Year 3, I might as well create a personal project about it. 

## Why this project?
Even though I have played Tetris for so long, I have barely improved. Even funnier, I have learned some methods to improve in Tetris but I struggle to adapt to them. So why not make an all-powerful AI that will play for me to create some perfect gameplay?

## What inspired me?
I was inspired to make this from the Youtuber Stuff Made Here, I have been watching videos for a while too, and his dedication and creativity from starting the project to refining his project has always been so awe-spiring. His videos are so cool, and I wish I can create Youtube videos like him, too. 

# Goals
I want to implement a Tetris AI that will ideally play with great placements and great foresight. There are Tetris battles on Tetr.io, and this will create `garbage`, making the battle quite intense. I will also simulate it like a human playing it, with some ideal placements I guess, the details are not super planned out but I will track my progress along the way.

## Goal 1: Godly Foresight
To start out, I want it to place pieces with great foresight. I will make it not use the hold button, so it is forced to play the pieces with consideration of the next 5 pieces it can see. This way, it can at least place pieces well without creating holes in the placement.

## Goal 2: Idealistic placement
With calculated placements, I want it to place with least key movements as possible. In Tetr.io, it is ideal to place a piece in 3 moves, with hard drop being the 3rd move (I think?).
These three moves are done as followed:
1. Rotate once (clock-wise, counter-clock-wise, 180 degrees),
2. Move once (left one, right one, border left, border right),
3. Hard drop.

For reference, my best personal record for fastest clearing 40 lines (at 37.117 seconds) has a 3.490 keys per piece. The world record (at 13.430 seconds!) has a 3.030 keys per piece. 

## Goal 3: Worthy opponent
After both calculated and efficient placement, I want to make it also manage garbage, attack, and defense.

## Goal 4: Undefeated champ
See it play! I don't know, I haven't thought this far. Hopefully I can even get this far actually.

# Progress checker

## Goal 0: Presets _(Work-In-Progress)_

I forgot to mention this! Before I make any of this Tetris AI, I have to first create the game. This includes rotation mechanics, how the tetrominoes are generated, and more!

1. Task 1: Generate Tetraminoes
2. Task 2: Rotate Tetramino
3. Task 3: *coming soon...*


## Goal 1: Godly Foresight _(To Be Implemented)_
