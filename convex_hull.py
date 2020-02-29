from functools import cmp_to_key

# 凸包算法
def isin(pi,pj,pk,p0):
    x1 = float(p0[0])
    x2 = float(pj[0])
    x3 = float(pi[0])
    x4 = float(pk[0])
    y1 = float(p0[1])
    y2 = float(pj[1])
    y3 = float(pi[1])
    y4 = float(pk[1])

    k_j0=0
    b_j0=0
    k_k0=0
    b_k0=0
    k_jk=0
    b_jk=0
    perpendicular1=False
    perpendicular2 = False
    perpendicular3 = False

    if x2 - x1 == 0:
        t1=(x3-x2)*(x4-x2)
        perpendicular1=True
    else:
        k_j0 = (y2 - y1) / (x2 - x1)
        b_j0 = y1 - k_j0 * x1
        t1 = (k_j0 * x3 - y3 + b_j0) * (k_j0 * x4 - y4 + b_j0)

    if x4 - x1 == 0:
        t2=(x3-x1)*(x2-x1)
        perpendicular2=True
    else:
        k_k0 = (y4 - y1) / (x4 - x1)
        b_k0 = y1 - k_k0 * x1
        t2 = (k_k0 * x3 - y3 + b_k0) * (k_k0 * x2 - y2 + b_k0)

    if x4 - x2 == 0:
        t3=(x3-x2)*(x1-x2)
        perpendicular3 = True
    else:
        k_jk = (y4 - y2) / (x4 - x2)
        b_jk = y2 - k_jk * x2
        t3 = (k_jk * x3 - y3 + b_jk) * (k_jk * x1 - y1 + b_jk)

    if (k_j0 * x4 - y4 + b_j0)==0 and (k_k0 * x2 - y2 + b_k0)==0 and  (k_jk * x1 - y1 + b_jk)==0 :
          t1=-1
    if perpendicular1 and perpendicular2 and perpendicular3:
          t1=-1

    return t1,t2,t3

def force(lis, n):
    if n==3:
        return  lis
    else:
        lis.sort(key = lambda x: x[1])
        p0=lis[0]
        lis_brute=lis[1:]
        temp=[]
        for i in range(len(lis_brute)-2):
            pi=lis_brute[i]
            if i in temp:
                continue
            for j in range(i+1,len(lis_brute) - 1):
                pj=lis_brute[j]
                if j in temp:
                    continue
                for k in range(j + 1, len(lis_brute)):
                    pk=lis_brute[k]

                    if k in temp:
                        continue
                    (it1,it2,it3)=isin(pi,pj,pk,p0)
                    if it1>=0 and it2>=0 and it3>=0:
                        if i not in temp:
                           temp.append(i)
                    (jt1,jt2,jt3)=isin(pj,pi,pk,p0)
                    if jt1>=0 and jt2>=0 and jt3>=0:

                        if j not in temp:
                           temp.append(j)

                    (kt1, kt2, kt3)=isin(pk, pi, pj, p0)
                    if kt1 >= 0 and kt2 >= 0 and kt3 >= 0:

                        if k not in temp:
                            temp.append(k)

        lislast=[]
        for coor in lis_brute:
            loc = [i for i, x in enumerate(lis_brute) if all(x == coor)]
            for x in loc:
                ploc = x
            if ploc not in temp:
                lislast.append(coor)

        lislast.append(p0)
        res = []
        for item in lislast:
            res.append([item[0], item[1]])
        return res

def cmp(a, b, center):
    if a[0] >= 0 and b[0] < 0:
        return True
    if a[0] == 0 and b[0] == 0:
        return a[1] > b[1]
    det = int((a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1]))
    if det < 0:
        return True
    if det > 0:
        return False
    d1 = float((a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1]))
    d2 = float((b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1]))
    return d1 > d2

def mySort(arr, center):
    for i in range(0, len(arr) - 1):
        for j in range(0, len(arr) - i - 1):
            if arr[j][0] > arr[j + 1][0]:
                tmp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = tmp

    for i in range(0, len(arr) - 1):
        for j in range(0, len(arr) - i - 1):
            #print(i, j, j + 1)
            if cmp(arr[j], arr[j + 1], center):
                tmp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = tmp