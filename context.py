symbols = ['+', '-', '/', '=', '\pm', '(', ')', '(', '[', ']', '{', '}']


def toLatex(exp):
    res = ''
    parent = 0
    levelsY = []
    for j in range(len(exp)):
        if(j > 0):
            if(isSameY(exp[j-1], exp[j])):
                res = res
            elif(isExponent(exp[j-1], exp[j])):
                res = res+'^{'
                levelsY.append(exp[j-1])
                parent += 1
            elif(isLowered(exp[j-1], exp[j])):
                res = res+'_{'
                levelsY.append(exp[j-1])
                parent += 1
            else:
                while (len(levelsY) > 0 and not isSameY(exp[j], levelsY[len(levelsY)-1])):
                    res = res+'}'
                    parent -= 1
                    levelsY.pop(len(levelsY)-1)
                if(len(levelsY) > 0):
                    res = res+'}'
                    parent -= 1
                    levelsY.pop(len(levelsY)-1)

        if (exp[j]['character'] == 'frac'):
            res += '\\frac'
            res += '{'+toLatex(exp[j]['over'])+'}'
            res += '{'+toLatex(exp[j]['bottom'])+'}'
        elif (exp[j]['character'] == 'sqrt'):
            res += '\sqrt'
            res += '{'+toLatex(exp[j]['bottom'])+'}'
        else:
            res += exp[j]['character']
    while (parent > 0):
        res = res+'}'
        parent -= 1
    return res


def findOver(e, j):
    res = []
    delete = []
    for k in range(j+1, len(e)):
        if(k >= len(e)):
            break
        if(e[k]['x'] <= e[j]['x'] + e[j]['w'] and e[k]['y'] < e[j]['y'] and e[k]['x'] + e[k]['w'] > e[j]['x']):
            res.append(e[k])
            delete.append(k)
    delete = sorted(delete, reverse=True)
    # for k in delete:
    #e[k]['active'] = False
    return res


def findUnder(e, j):
    res = []
    delete = []
    for k in range(j+1, len(e)):
        if(k >= len(e)):
            break
        if(e[k]['x'] <= e[j]['x'] + e[j]['w'] and e[k]['y'] > e[j]['y'] and e[k]['x'] + e[k]['w'] > e[j]['x']):
            res.append(e[k])
            delete.append(k)
    delete = sorted(delete, reverse=True)
    # for k in delete:
    #e[k]['active'] = False
    return res


def findInner(e, j):
    res = []
    delete = []
    for k in range(j+1, len(e)):
        if(k >= len(e)):
            break
        if(e[k]['x'] <= e[j]['x'] + e[j]['w'] and e[k]['x'] > e[j]['x'] and e[k]['y'] > e[j]['y'] and e[k]['y'] <= e[j]['y'] + e[j]['h']):
            res.append(e[k])
            delete.append(k)
    delete = sorted(delete, reverse=True)
    # for k in delete:
    #e[k]['active'] = False
    return res


def isSameX(ch, oth):
    ch_center_x = ch['center'][0]
    oth_center_x = oth['center'][0]
    if abs(ch_center_x - oth_center_x) < 30:
        return True
    else:
        return False


def isInside(ch, oth):
    if oth['y'] < (ch['y']+ch['h']) and oth['x'] > ch['x'] and (oth['x']+oth['w']) - (ch['x']+ch['w']) < 10:
        return True
    else:
        return False


def isNumer(ch, oth):
    if (oth['y']+oth['h']) < ch['y'] and oth['x'] - ch['x'] > -10 and (oth['x']+oth['w']) - (ch['x']+ch['w']) < 10:
        return True
    else:
        return False


def isDenom(ch, oth):
    if oth['y'] > (ch['y']+ch['h']) and oth['x'] - ch['x'] > -10 and (oth['x']+oth['w']) - (ch['x']+ch['w']) < 10:
        return True
    else:
        return False


def isExponent(ch, oth):
    ch_center = ch['center'][1]
    oth_center = oth['center'][1]
    ch_center_x = ch['center'][0]
    if oth_center < ch_center and oth['x'] > ch_center_x and ch['character'] not in symbols:
        return True
    else:
        return False


def isLowered(ch, oth):
    ch_center = ch['center'][1]
    oth_center = oth['center'][1]
    ch_center_x = ch['center'][0]
    if oth_center > ch_center and oth['x'] > ch_center_x and oth_center < ch['y'] + ch['h'] + 5 and ch['character'] not in symbols:
        return True
    else:
        return False


def isSameY(ch, oth):
    ch_center = ch['center'][1]
    oth_center = oth['center'][1]
    if abs(ch_center - oth_center) < ch['h']/4 or abs(ch_center - oth_center) < oth['h']/4:
        return True
    else:
        return False


def area(symbol):
    return (symbol['w'] - symbol['x']) * (symbol['h'] - symbol['y'])


def updateEqual(symbol, s1, symbol_list, im, i):
    new_x = min(symbol['x'], s1['x'])
    new_y = min(symbol['y'], s1['y'])
    new_xw = max(symbol['w'], s1['w'])
    new_yh = max(symbol['h'], s1['h'])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)),
                  "=", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)


def updateDivision(symbol, s1, s2, symbol_list, im, i):
    new_x = min(symbol['x'], s1['x'], s2['x'])
    new_y = min(symbol['y'], s1['y'], s2['y'])
    new_xw = max(symbol['w'], s1['w'], s2['w'])
    new_yh = max(symbol['h'], s1['h'], s2['h'])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)),
                  "div", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)


def updateDots(symbol, s1, s2, symbol_list, im, i):
    new_x = min(symbol['x'], s1['x'], s2['x'])
    new_y = min(symbol['y'], s1['y'], s2['y'])
    new_xw = max(symbol['w'], s1['w'], s2['w'])
    new_yh = max(symbol['h'], s1['h'], s2['h'])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)),
                  "dots", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)


def updateI(symbol, s1, symbol_list, im, i):
    new_x = min(symbol['x'], s1['x'])
    new_y = min(symbol['y'], s1['y'])
    new_xw = max(symbol['w'], s1['w'])
    new_yh = max(symbol['h'], s1['h'])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)),
                  "i", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)


def updatePM(symbol, s1, symbol_list, im, i):
    new_x = min(symbol['x'], s1['x'])
    new_y = min(symbol['y'], s1['y'])
    new_xw = max(symbol['w'], s1['w'])
    new_yh = max(symbol['h'], s1['h'])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)),
                  "pm", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)


def updateBar(symbol, symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "bar", x, y, xw, yh)
    symbol_list[i] = new_symbol


def updateFrac(symbol, symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "frac", x, y, xw, yh)
    symbol_list[i] = new_symbol
