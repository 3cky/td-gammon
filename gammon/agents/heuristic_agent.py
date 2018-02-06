import numpy as np

from ..game import Game


class HeuristicAgent(object):

    WHITE = 0
    BLACK = 1

    INDEX_RIGHT_BOT = 0
    INDEX_LEFT_BOT = 11
    INDEX_LEFT_TOP = 12
    INDEX_RIGHT_TOP = 23

    WHITE_HEAD_INDEX = INDEX_RIGHT_TOP
    BLACK_HEAD_INDEX = INDEX_LEFT_BOT

    WHITE_HOME_S = 0  # start white checker's home index
    WHITE_HOME_E = 5  # end white checker's home index

    BLACK_HOME_S = 12  # start black checker's home index
    BLACK_HOME_E = 17  # end black checker's home index

    # Invalid value code
    INVALID_VALUE = -1

    # Number of cells (point) on the 1 quarter of board
    NUM_POINTS = 6

    # ********************* Weights constants *****************************
    #  Weights of checkers in the barrier for different quadrant
    W_CLOSED_I_B = 4  # indexes [23-18] begin of the game - when we have checkers on the head
    W_CLOSED_I_3_B = 8  # [23-18] when closed more 3 checkers
    W_CLOSED_I_4_B = 10  # [23-18] when closed more 4 checkers
    W_CLOSED_I = 2  # indexes [23-18]
    W_CLOSED_I_3 = 4  # [23-18] when closed more 3 checkers
    W_CLOSED_II = 3  # indexes [17-12]
    W_CLOSED_II_3 = 4  # [17-12] when closed more 3 checkers
    W_CLOSED_III_B = 6  # indexes [11-6] begin of the game
    W_CLOSED_III_3_B = 10  # indexes [11-6]
    W_CLOSED_III = 4  # indexes [11-6]
    W_CLOSED_III_3 = 4  # indexes [11-6]
    W_CLOSED_IV = 6  # indexes [5-0]
    W_CLOSED_IV_3 = 9  # indexes [5-0]
    #  Additional weight for barrier consists 6 checkers and more
    W_CLOSED_6_TOP = 70  # when closed top of board [11-0]
    W_CLOSED_6_BOT = 10  # when closed bottom of board [23-12]
    #  Amount of home checkers weight
    W_HOME_AMOUNT = 9
    #  When we have checkers in 1st quadrant
    W_HOME_AMOUNT_1 = 5
    #  In the start of game we shouldn't came to home
    W_NOT_HOME_AMOUNT = -3
    #  Amount of my out checker
    W_OUT_AMOUNT = 50
    #  Weight of home closed points
    W_CLOSED_HOME_BEGIN = 1    # при начале игры
    W_CLOSED_HOME = 5
    W_CLOSED_HOME_END = 2     # в конце игры, когда опп уже увёл свои шашки
    #  Coefficient for checkers distance from end
    W_MAX_INDEX = -1
    W_MAX_INDEX_1 = -3
    W_MIN_INDEX = -1
    #  Weight for amount of closed points on the top and bottom of the board
    W_CLOSED_POINTS_TOP = 7
    W_CLOSED_POINTS_BOT = 4
    #  Weight of I-st quadrant of checkers if we have checkers on the Head
    W_CLOSED_POINTS_I = 9
    #  Checkers on the head
    W_ON_HEAD = -5
    W_ON_HEAD_BEGIN = -7  # при начале игры
    #  Distance of checkers to home
    W_HOME_DISTANCE = -1
    #  Amount of my checkers on the top of the board
    TOP_CHECKERS_SEPARATE = 9

    def __init__(self, player):
        self.player = player
        self.name = 'Heuristic'

    def choose_action(self, actions, game):
        actions = list(actions)

        weights = []

        for a in actions:
            game.take_action(a, self.player)
            weights.append(self.evaluate_board(game))
            game.undo_action(a, self.player)

        return actions[np.argmax(weights)]

#     /**
#      * Evaluate current checkers layout on the board at dependence from current
#      * moving user.
#      */
    def evaluate_board(self, game):
        localBoard = game.grid
        # get players color
        myColor = game.players.index(self.player)
        oppColor = 1 - myColor
        #  Max index of my checkers in absolute indexes
        maxMyIndex = 0
        #  Min index of my checkers in absolute indexes (не учитывая шашки в домике)
        minMyIndex = 99  # берём не валидное значение, т.к. при поиске не учитываются шашки в домике!
        #  Max index of opp's checkers
        maxEnemyIndex = 0
        #  Amount of our out checkers
        myOutCheckers = len(game.off_pieces[self.player])
        #  Amount of home checkers
        homeAmount = myOutCheckers
        homeAmountWeight = self.W_HOME_AMOUNT
        #  Amount of closed points in my home
        closedHomePoints = 0
        #  Amount of my checkers in current po(translated for black!)
        # myCheckersAmount
        #  Amount of enemy's checkers in current po
        # enemyCheckersAmount
        #  Amount of my checkers on the head
        # myHead
        # Total amount of closed points of my checkers
        # on the top and bottom of the board
        closedPointsTop = 0
        closedPointsTop3 = 0          # закрытые поинты в 3ем квадранте (пипсы 6-11)
        closedPointsTopTotal = 0   # with checkers amount in each point
        closedPointsBot = 0
        closedPointsI = 0
        #  общее кол-во шашек в первой четверти, в итоге - вес!
        totalPointsI = 0
        #  Total evaluation of the board's layout
        # evaluation = 0
        #  Barriers' weights
        blocking = 0
        #  Block's start and end indexes
        # bStart
        # bEnd
        #  Amount of opponent's checkers in their 1st quadrant
        checkersAt1 = -1
        #  Amount of opponent's checkers in their 3d and 4th quadrants (top of the board for color)
        oppTopCheckers = 0
        #  Distance to home of our checkers
        distance = 0
        # Барьеры оппонента за моим maxIndex с учётом пустых ячеек, т.е. если он потенциально
        # может мне заблокировать шашки. Вносит отрицательный вес.
        enemyBlock = 0

        # count amount of checkers on the my head
        myHead = self.num_color_checkers(localBoard, self.head_idx(myColor), myColor)
        # going through the board and obtain everything that we can obtain in one loop
        for i in range(23, -1, -1):
            # получаем кол-во своих или противника шашек в рассматривоваемом поинте
            myCheckersAmount = self.num_color_checkers(localBoard, i, myColor)
            enemyCheckersAmount = self.num_color_checkers(localBoard, i, oppColor)
            if myCheckersAmount > 0:
                if self.trans_pos(myColor, i) <= 11:
                    closedPointsTop += 1        # 3d, 4th quadrants
                    if self.trans_pos(myColor, i) > 5:
                        closedPointsTop3 += 1
                    closedPointsTopTotal += myCheckersAmount
                else:    # 1st, 2nd quadrants
                    # bottom of the board:
                    closedPointsBot += 1
                    # accumulate additional weight for 1st quadrant closed points
                    if self.trans_pos(myColor, i) <= 23 and self.trans_pos(myColor, i) >= 18:
                        closedPointsI += 1
                        totalPointsI += myCheckersAmount
                if self.trans_pos(myColor, i) > maxMyIndex:
                    maxMyIndex = self.trans_pos(myColor, i)   # hold max index
                # не рассматриваем (исключаем) шашки в домике!
                if self.trans_pos(myColor, i) < minMyIndex and not self.is_home_index(i, myColor):
                    minMyIndex = self.trans_pos(myColor, i)  # hold min index
                # count distance to home if checkers beyond home
                if not self.is_home_index(i, myColor):
                    distance += (self.trans_pos(myColor, i) - 5) * myCheckersAmount
                else:
                    # home area...
                    closedHomePoints += 1
                    homeAmount += myCheckersAmount
            else:
                if enemyCheckersAmount > 0 and self.trans_pos(oppColor, i) > maxEnemyIndex:
                    maxEnemyIndex = self.trans_pos(oppColor, i)  # hold max index for opponent checkers
                # count opponent's checkers in 3d and 4th quadrants (top of the board)
                if self.trans_pos(oppColor, i) <= 11:
                    oppTopCheckers += enemyCheckersAmount
        # === end for i by whole board ===

        # Если на верху доски мало шашек, и на голове больше 3, то пока в домик не заводим
        if maxMyIndex > 17 and (closedPointsTopTotal < self.TOP_CHECKERS_SEPARATE and myHead > 5):
            homeAmountWeight = self.W_NOT_HOME_AMOUNT
        else:
            # у меня есть шашки в первой четверти
            if maxMyIndex > 17:
                homeAmountWeight = self.W_HOME_AMOUNT_1
        # Удалим из подсчёта шашки, которые находятся за maxEnemyIndex, т.к. они
        #  реально уже ничего не закрывают
        if maxEnemyIndex < 23 and maxEnemyIndex >= 12:
            for i in range(maxEnemyIndex + 1, 24):
                if self.num_color_checkers(localBoard, self.trans_pos(oppColor, i), myColor) > 0:
                    closedPointsTop -= 1
                    if self.trans_pos(myColor, i) > 5 and self.trans_pos(myColor, i) <= 11:
                        closedPointsTop3 -= 1

        # найдём кол-во шашек оппа в их первой четверти
        if maxEnemyIndex > 17:
            checkersAt1 = 0
            for i in range(18, 24):
                checkersAt1 += self.num_color_checkers(localBoard, self.trans_pos(oppColor, i), oppColor)
        # find all barriers on the field...
        for i in range(23, 0, -1):
            bStart = self.INVALID_VALUE
            bEnd = self.INVALID_VALUE
            if self.num_color_checkers(localBoard, self.trans_pos(myColor, i), myColor) > 0 \
                    and maxEnemyIndex > self.trans_pos(oppColor, self.trans_pos(myColor, i)):
                # look at next cell
                if self.trans_pos(myColor, i) > 0 and \
                        self.num_color_checkers(localBoard, self.trans_pos(myColor, i-1), myColor) > 0:
                    bStart = i
                    bEnd = i-1
                    # find end of barrier
                    j = bEnd - 1
                    # не рассматриваем продолжение забора в 1ой четверти оппа
                    while j >= 0 and bEnd != 12 and bStart != 12 \
                            and self.num_color_checkers(localBoard, self.trans_pos(myColor, j), myColor) > 0:
                        j -= 1
                        bEnd -= 1
                    # initialize i index after found barrier
                    if bEnd >= 0:
                        i = bEnd
                    # calculate block weight...
                    blocking += (bStart - bEnd + 1) * \
                        self.quadrant_weight(bStart, bEnd, maxMyIndex, maxEnemyIndex,
                                             checkersAt1, oppTopCheckers)
                    # Если блокировка на голове, то надо проверить, что может мы
                    #  закрыли ещё и поля с 0-го и далее, т.е. у себя в домике
                    if bStart == 23:
                        k = 0
                        while self.num_color_checkers(localBoard, self.trans_pos(myColor, k), myColor) > 0:
                            blocking += self.W_CLOSED_IV
                            k += 1
                        if (bStart - bEnd + 1 + k) >= self.NUM_POINTS:
                            blocking += self.W_CLOSED_6_TOP
                        elif (bStart - bEnd + 1) + k > 3:
                            # if closed more 3 points
                            blocking += self.W_CLOSED_IV_3
                    # check up if we closed 6 or more points
                    if (bStart - bEnd + 1) >= self.NUM_POINTS:
                        blocking += self.quadrant_6_closed_weight(bStart, bEnd)

        # find barriers for opponent
        continuousBlockExist = False  # непрерывный блок из 5ти шашек (и более)!
        for i in range(23, 0, -1):
            if self.num_color_checkers(localBoard, self.trans_pos(oppColor, i), oppColor) > 0 \
                    and maxMyIndex > self.trans_pos(myColor, self.trans_pos(oppColor, i)):
                pipsWeight = 2    # есть шашка в поинте
                busyPips = 1
                continBlock = 1   # непрерывный блок (длина)
                # ищем барьеры оппонента
                bStart = i
                bEnd = i
                # find end of barrier
                j = bEnd - 1
                while j >= 0 and self.trans_pos(myColor, self.trans_pos(oppColor, j)) < maxMyIndex \
                        and (self.num_color_checkers(localBoard, self.trans_pos(oppColor, j), oppColor) > 0 \
                             or self.num_checkers(localBoard, self.trans_pos(oppColor, j)) == 0):
                    # для чёрных нет смысла смотреть блокировки белых, начинающиеся на верху и
                    #  продолжение их в первой четверти.
                    if myColor == self.BLACK and bStart >= 12 and j < 12:
                        break
                    pipsWeight += 1
                    if self.num_color_checkers(localBoard, self.trans_pos(oppColor, j),
                                                oppColor) > 0:
                        # увеличиваем вес пипса, если в нём есть шашка, а не пустой он
                        pipsWeight += 1
                        busyPips += 1
                        continBlock += 1
                    else:
                        continuousBlockExist = (continBlock >= 5 or continuousBlockExist)
                        continBlock = 0
                    j -= 1
                    bEnd -= 1
                bLen = bStart-bEnd+1
                if bLen > 1:
                    if bLen < 6 and (bLen - busyPips > 1 or bLen == 2):
                        enemyBlock -= pipsWeight
                    else:
                        if busyPips > 3 and busyPips > (bLen - busyPips):
                            enemyBlock -= 4*pipsWeight    # могут заблокировать!
                        else:
                            if busyPips * 2 < bLen:
                                enemyBlock -= pipsWeight    # не так страшно...
                            else:
                                enemyBlock -= 2*pipsWeight  # чуть хуже...
                        # уже 5 поинтов подряд закрыты и могут закрыть 6ой!
                        if continuousBlockExist:
                            enemyBlock -= 15
                    i = bEnd - 1

        # пересчитаем некоторые веса и переменные в зависисмости от определённых ситуаций

        # Проверяем,что опп уже перевёл много своих шашек наверх (воспользуемся его шашками, т.к.
        #  их позиции не меняются в процессе рассмотрения моих вариантов ходов)
        if oppTopCheckers > 10:
            # ухудшим вес шашек в первой четверти... - пора сваливать!
            totalPointsI = -4 * totalPointsI
        else:
            totalPointsI = 0

        # проверим значение моего минимального индекса, если он больше 23, занчит все шашки уже
        #  в домике и учитывать его не надо
        if minMyIndex > 23:
            minMyIndex = 0
        # пересчитаем вес для закрытых поинтов в домике
        wClosedHome = self.W_CLOSED_HOME
        if myHead > 5:
            # считаем как начало игры
            wClosedHome = self.W_CLOSED_HOME_BEGIN
        else:
            # если опп уже увёл свои шашки дальше твоего домика, то нет смысла блокировать дом
            #  и делать такой большой вес закрытого поинта в доме! Он должен быть не нулевым
            #  лишь для того, чтобы закрыть поинт было лучше, чем оставить его открытытм и всё.
            #  Т.е. чтобы ИИ старался всё-таки не отсавлять пустых поинтов в домике, при прочих
            #  равных, чтобы потом легче было выбрасывать шашки
            if maxEnemyIndex < 14:
                wClosedHome = self.W_CLOSED_HOME_END
        # добавим вес шашек в нашей 3ей четверти при начале игры
        if myHead > 5 and homeAmount < 5:
            closedPointsTop3 = self.W_CLOSED_III_B * closedPointsTop3
        else:
            closedPointsTop3 = 0
        # учтём ещё такой фактор для вывода последний шашки, когда уже нет шашек на голове
        #  чтобы ИИ не оставлял их, а при ходе отдавал предпочтение именно последней шашке.
        #  Т.е. усилю вес расстояния между головой и последней шашкой - чем больше, тем лучше!
        lastCheckerWeight = (23 - maxMyIndex)
        if oppTopCheckers > 10 or maxMyIndex > 18:
            # под самый конец игры важенне уводить дальние шашки!!!
            lastCheckerWeight *= 3
        else:
            lastCheckerWeight *= 2

        # calculating final evaluation...
        return homeAmountWeight * homeAmount \
            + wClosedHome * closedHomePoints \
            + self.W_OUT_AMOUNT * myOutCheckers \
            + (self.W_MAX_INDEX_1 * maxMyIndex if (maxEnemyIndex < 13)
               else self.W_MAX_INDEX * maxMyIndex) \
            + lastCheckerWeight \
            + ((self.W_ON_HEAD_BEGIN if myHead > 8 else self.W_ON_HEAD) * myHead) \
            + (0 if (maxEnemyIndex < 15) else self.W_CLOSED_POINTS_TOP * closedPointsTop) \
            + (0 if (maxEnemyIndex <= 5) else self.W_CLOSED_POINTS_BOT * closedPointsBot) \
            + closedPointsTop3 \
            + self.W_CLOSED_POINTS_I * closedPointsI \
            + totalPointsI \
            + blocking \
            + self.W_HOME_DISTANCE * distance \
            + (self.W_MIN_INDEX * minMyIndex if oppTopCheckers == 15
               else (0 if myHead > 7 else -minMyIndex)) \
            + enemyBlock

    def trans_pos(self, color, i):
        return i if color == self.WHITE else Game.flip_position(i)

    def head_idx(self, color):
        return self.WHITE_HEAD_INDEX if color == self.WHITE else self.BLACK_HEAD_INDEX

    def is_home_index(self, index, color):
        if color == self.WHITE:
            start = self.WHITE_HOME_S
            end = self.WHITE_HOME_E
        else:
            # FIXME is different for Backgammon
            start = self.BLACK_HOME_S
            end = self.BLACK_HOME_E
        return index >= start and index <= end

    def num_color_checkers(self, board, index, color):
        return len(board[index]) if (board[index] and board[index][0] == Game.PLAYERS[color]) \
            else 0

    def num_checkers(self, board, index):
        return len(board[index])

#     /**
#      * Calculate the middle point of barrier and return appropriate weight.
#      * @param start - start index of barrier in absolute indexes;
#      * @param end - end index of barrier in absolute indexes;
#      * @param maxBlackIndex - max value of index of black checkers;
#      * @param maxWhiteIndex - max index of white checkers;
#      * @param white1 - amount of white checkers in their 1st quadrant;
#      * @param white34 - amount of white checkers in their 3d and 4th quadrant
#      * @return appropriate quadrant's weight.
#      */
    def quadrant_weight(self, start, end, maxBlackIndex, maxWhiteIndex, white1, white34):
        idx = ((start - end) >> 1) + end
        # если забор из 2ух шашек, то индекс берём по старт индексу!
        if start - end == 1:
            idx += 1
        # count barrier's length
        blen = start - end + 1
        if idx >= self.INDEX_RIGHT_TOP - self.NUM_POINTS + 1:
            # не имеет смысл закрывать первую четверть, если белые уже пришли,
            # заграждение мало (меньше 5) и у чёрных не осталось шашек за заграждением
            if maxWhiteIndex < 12 and blen < 5 and maxBlackIndex <= start:
                return 0
            if start < maxBlackIndex and white34 < 8:
                if blen > 4:
                    return self.W_CLOSED_I_4_B
                return self.W_CLOSED_I_3_B if blen > 3 else self.W_CLOSED_I_B
            # если у нас есть ещё шашки сзади, например, на голове, то закрывать здесь
            # поинты важнее, чем если своих шашек за загрождением уже нет.
            if white34 > 8 and maxBlackIndex > 17 and maxBlackIndex != start:
                return self.W_CLOSED_I_3
            return self.W_CLOSED_I_3 if blen > 3 else self.W_CLOSED_I
        if idx >= self.INDEX_LEFT_TOP:
            # не открываем пока закрытый домик!
            if blen == 4 and maxBlackIndex <= start and start == 17:
                return 4
            # не имеет смысл закрывать первую четверть, если белые уже пришли,
            #  заграждение мало (меньше 5) и у чёрных не осталось шашек за заграждением
            if maxWhiteIndex < 12 and blen < 5 and maxBlackIndex <= start:
                return 2
            if start < maxBlackIndex and white34 > 10:
                return 5 if blen > 3 else 4
            return self.W_CLOSED_II_3 if blen > 3 else self.W_CLOSED_II
        if idx >= self.INDEX_RIGHT_BOT + self.NUM_POINTS:
            if white1 < 4:
                return self.W_CLOSED_III_3 if blen > 3 else self.W_CLOSED_III
            return self.W_CLOSED_III_3_B if blen > 3 else self.W_CLOSED_III_B
        if idx >= self.INDEX_RIGHT_BOT:
            # если блок всего из двух шашек, то делаем вес поменьше
            return self.W_CLOSED_IV_3 if blen > 3 else (3 if blen == 2 else self.W_CLOSED_IV)
        return self.W_CLOSED_I

#     /**
#      * Calculate the middle point of barrier and return appropriate additional
#      * weight of quadrant when closed 6 or more points.
#      * @param start - start index of barrier in absolute indexes
#      * @param end - end index of barrier in absolute indexes.
#      * @return appropriate additional quadrant's weight.
#      */
    def quadrant_6_closed_weight(self, start, end):
        idx = ((start - end) >> 1) + end
        if idx >= self.INDEX_LEFT_TOP:
            return self.W_CLOSED_6_BOT
        if idx >= self.INDEX_RIGHT_BOT:
            return self.W_CLOSED_6_TOP
        return self.W_CLOSED_6_BOT
