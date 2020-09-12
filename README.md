# rl-huy
This is my solution for RLCOMP2020 reaching top 65. We used many approaches and different techniques including variations  of deep q learning, imitation learning. 
After a month of testing and researching, I found that the key component of initial agent is the use of CNN. Because of difference between maps, so using 2D state would result a 
robust performance and generalize well across maps. Secondly, I tested many smaller combinations of RAINBOW algorithm, and the conbination including double deep q learning,
prioritized experience memory, n-step learning give the most promising metrics. I aslo wrote a code that implement the learning from human demonstation paper. My teamates coded a
heuristic bot based on A* algorithm then I used it as a demonstration generator instead of a human. However, due to lack of memory so this approach hasn't evalutated but I 
considered it as very promising approach.
