#How this whole things works:
#The most central function is called workhorse (line 158)
#workhorse assumes the previous iteration ended with a phi step and begins with a phi_inverse step
#There are two functions that do the intersecting, part_int_backward and part_int_forward
#_backward means an phi_inverse step, forward means a phi step
#checker_test checks for redundancies by running through a tuple of lineage values
#part_int_forward and part_int_backward are basically workarounds that account for the fact that checker_test was designed for forward steps. part_int backward flips the inputs before feeding them to checker_test and makes the appropriate changes to the lineage creation steps



def rint_check_firstgen(R1,R2,index1,index2):
        lineage = [index1,index2]
        R = []
        xm1 = n(R1.rec[0][0])
        xM1 = n((R1.rec[0]+R1.rec[1]*s+R1.rec[2]*u)[0])
        ym1 = n((R1.rec[0]+R1.rec[1]*s)[1])
        yM1 = n((R1.rec[0]+R1.rec[2]*u)[1])
        xm2 = n(R2.rec[0][0])
        xM2 = n((R2.rec[0]+R2.rec[1]*s+R2.rec[2]*u)[0])
        ym2 = n((R2.rec[0]+R2.rec[1]*s)[1])
        yM2 = n((R2.rec[0]+R2.rec[2]*u)[1])
        for (i,j) in itertools.product([floor(xm1-xM2)+1..ceil(xM1-xm2)-1], [floor(ym1-yM2)+1..ceil(yM1-ym2)-1]):
            result = rint0(R1.rec, radd((i, j), R2.rec))
            recksult = recker(result,lineage)
            R += recksult
        return R


def part_int_firstgen(partition1,partition2):
        rectangles = []
        mema = []
        rowcount = 0
        for R1 in partition1:
            columncount = 0
            row = []
            for R2 in partition2:
                Q = rint_check_firstgen(R1,R2,rowcount,columncount)
                if Q!=[]:
                    rectangles = rectangles + Q
                    row.append(1)
                else:
                    row.append(0)
                columncount += 1
            mema.append(row)
            rowcount += 1
        return (rectangles,mema)



#Definitions:
#Couple - An image of the original parititon under some degree of phi
#Ancestor - An element of a couple
#Lineage - A tuple of ancestors for a rectangle, goes from couple of lowest degree to couple of highest degree

class Reck:
    def __init__(self,rectangle,lineage):
        self.rec = rectangle
        self.line = lineage

#Redundancy to check for:
#Type 1 - Two rectangles can't overlap because they have different ancestors in one of the couples
#Type 2 - Two rectangles can't overlap because they're have ancestors from different couples that we know don't overlap


#The phi functions work just like before except they carry lineage along
#Note that this means the lineage tuple of the output will be identical, but it points to different couples. Other functions will be written with this in
#    mind

def phi_check(rectangle):
    return(Reck(phi(rectangle.rec),rectangle.line))

def phi_inv_check(rectangle):
    return(Reck(phi_inv(rectangle.rec),rectangle.line))

def phi_part_check(partition):
    return[phi_check(rectangle) for rectangle in partition]

def phi_inv_part_check(partition):
    return [phi_inv_check(rectangle) for rectangle in partition]


#Runs through lineage and checks for type 1 redundancies
#Returns true if there is no type 1 redudancies, false if there are
#Specifically designed to be applied to a pair of the form (rec1, phi(rec2)) where rec1,rec2 are elements of our partition
#However I think flipping the inputs to the form (phi_inv(rec1),rec2) will make it work in the other direction
def checker_test(rectangle_one,rectangle_two):
    ancestry_one = rectangle_one.line
    ancestry_two = rectangle_two.line
    checks = len(ancestry_one)
    counter = 1
    diditwork = True
    while checks-counter > 0 :
        related = (ancestry_one[counter] == ancestry_two[counter-1])
        counter += 1
        if related == True:
            pass
        else:
            diditwork = False
            break
    return diditwork

#Updates the lineages of a list of rectangles
def ancestry_dot_com(rectangles,documents):
    for rectangle in rectangles:
        rectangle.line = documents

#Recker takes in non-object rectangle and turns it into an object rectangle with documents as the lineage
def recker(rectangles,documents):
    if rectangles != []:
        return [Reck(rectangles,documents)]
    else:
        return []

#Intersects two rectangles
#If checker_test returns true it runs rint
#Attaches the lineage of rectangle 1 with the last item in rectangle 2's lineage appended to it
#    Previous step should effectively build an accurate map to all ancestors for new rectangles if moving in forward direction
#    TBD whether reversing inputs works for reversing direction
def rint_check(R1,R2):
    if checker_test(R1,R2) == True:
        lineage = R1.line + [R2.line[-1]]
        R=[]
        xm1 = n(R1.rec[0][0])
        xM1 = n((R1.rec[0]+R1.rec[1]*s+R1.rec[2]*u)[0])
        ym1 = n((R1.rec[0]+R1.rec[1]*s)[1])
        yM1 = n((R1.rec[0]+R1.rec[2]*u)[1])
        xm2 = n(R2.rec[0][0])
        xM2 = n((R2.rec[0]+R2.rec[1]*s+R2.rec[2]*u)[0])
        ym2 = n((R2.rec[0]+R2.rec[1]*s)[1])
        yM2 = n((R2.rec[0]+R2.rec[2]*u)[1])
        for (i,j) in itertools.product([floor(xm1-xM2)+1..ceil(xM1-xm2)-1], [floor(ym1-yM2)+1..ceil(yM1-ym2)-1]):
            result = rint0(R1.rec, radd((i, j), R2.rec))
            recksult = recker(result,lineage)
            R += recksult
        return R
    else:
        return []

def rint_check_backward(R1,R2):
    if checker_test(R2,R1) == True:
        lineage = [R2.line[0]] + R1.line
        R=[]
        xm1 = n(R1.rec[0][0])
        xM1 = n((R1.rec[0]+R1.rec[1]*s+R1.rec[2]*u)[0])
        ym1 = n((R1.rec[0]+R1.rec[1]*s)[1])
        yM1 = n((R1.rec[0]+R1.rec[2]*u)[1])
        xm2 = n(R2.rec[0][0])
        xM2 = n((R2.rec[0]+R2.rec[1]*s+R2.rec[2]*u)[0])
        ym2 = n((R2.rec[0]+R2.rec[1]*s)[1])
        yM2 = n((R2.rec[0]+R2.rec[2]*u)[1])
        for (i,j) in itertools.product([floor(xm1-xM2)+1..ceil(xM1-xm2)-1], [floor(ym1-yM2)+1..ceil(yM1-ym2)-1]):
            result = rint0(R1.rec, radd((i, j), R2.rec))
            recksult = recker(result,lineage)
            R += recksult
        return R
    else:
        return []

#Should effectively run part_int when fed appropriate inputs for forward direction
def part_int_forward_matrix(partition1,partition2):
        rectangles = []
        mema = []
        for R1 in partition1:
            for R2 in partition2:
                Q = rint_check(R1,R2)
                if Q != []:
                    rectangles = rectangles + Q
                    mema.append(1)
                else:
                    mema.append(0)
        return (rectangles,matrix(len(partition1),len(partition2),mema))


#Works like part_int_forward except it flips the inputs when it executes rint_check
#    If flipping rint inputs does what I hope it does, this should run part_int in the other direction
#    Meet MAtrix made with rows being partition1 and columns being partition2


def part_int_backward_matrix(partition1,partition2):
        rectangles = []
        mema = []
        for R1 in partition1:
            for R2 in partition2:
                Q = rint_check_backward(R1,R2)
                if Q != []:
                    rectangles = rectangles + Q
                    mema.append(1)
                else:
                    mema.append(0)
        return (rectangles,matrix(len(partition1),len(partition2),mema))


#workhorse is going to functon as the main partitioning function, switch to showpony for the last generation in order to get an appropriate meet matrix
#Note that neither workhorse nor showpony use the meet matrices that they take as inputs, as checker_test is only currently checking for type 1 redundancies
#    Expanding checker_test to also check for type 2 redundancies would necessitate structural changes in these functions, though likely not large ones
def workhorse(partition,meet_matrix):
    post_inversion = part_int_backward_matrix(partition,phi_inv_part_check(partition))
    all_done = part_int_forward_matrix(post_inversion[0], phi_part_check(post_inversion[0]))
    return all_done

def get_key(reck):
    return reck.rec[3]

def find_threshold_radius(Rectangle):
    potential_thresholds = []
    left_most_corner = Rectangle[0]
#    print("Left most corner is ", left_most_corner)
    adjacent_corner_one = left_most_corner + (Rectangle[1]*s)
#    print("Lower adjacent corner is ",adjacent_corner_one)
    adjacent_corner_two = left_most_corner + (Rectangle[2]*u)
#    print("Upper adjacent corner is ",adjacent_corner_two)
    opposite_corner = left_most_corner + (Rectangle[1]*s) + (Rectangle[2]*u)
#    print("Opposite corner is ",opposite_corner)
    Rectangle_corners = [left_most_corner,adjacent_corner_one,adjacent_corner_two,opposite_corner]
    for (a,b) in itertools.product([-2..2],[-2..2]):
        corner_norm_distances = []
        for corner in Rectangle_corners:
            x = corner[0]
            y = corner[1]
            norm = Nm(((a-x),(b-y)))
            norm = abs(norm)
#            print("Here's the norm for a corner ",norm)
            corner_norm_distances.append(norm)
#        print("Here are the corner norm distances for one rectangle ",corner_norm_distances)
        one_potential_threshold = max(corner_norm_distances)
        potential_thresholds.append(one_potential_threshold)
#    print("Here are the potential thresholds- we pick the smallest among these to be the threshold radius ",potential_thresholds)
    threshold_radius = min(potential_thresholds)
    return(threshold_radius)

def showpony(partition,meet_matrix):
    post_inversion = part_int_backward_matrix(partition,phi_inv_part_check(partition))
    for reck in post_inversion[0]:
        reck.rec.append(find_threshold_radius(reck.rec))
    post_inversion[0].sort(key = get_key)
    all_done = part_int_forward_matrix(post_inversion[0], phi_part_check(post_inversion[0]))
    return (post_inversion[0],all_done[1])

def showpony_alt(partition,meet_matrix):
    post_inversion = part_int_backward_matrix(partition,phi_inv_part_check(partition))
    all_done = part_int_forward_matrix(post_inversion[0], phi_part_check(post_inversion[0]))
    return (post_inversion[0],all_done[1])

def draw_objects(partition):
    P = point((0,0))
    for m in partition:
        P = P+draw_nl(m.rec)+draw_boundary(m.rec)
    return P

def draw_objects_norm(partition):
    P = point((0,0))
    for m in partition:
        rectangle = m.rec[:3]
        P = P+draw_nl(rectangle)+draw_boundary(rectangle)
    return Pdef alt_normfinder(reck):
    "calculate actual norm that it takes to cover the rectangle with lattice points"
    rectangle = reck.rec
    results = []
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            results.append([abs(Nm(rectangle[0]-vector([x,y]))),vector((x,y))])
    results.sort(key = lambda result: result[0])
    top_four_points = results[:3]
    for point in top_four_points:
        max_dist = max_distance(reck,point[1])
        point.append(max_dist)
    top_four_points.sort(key = lambda top_four_points: top_four_points[2])
    return top_four_points[0][2]

def max_distance(reck,point):
    "returns norm such that the given rectangle is entirely included in the ball of the given point"
    R = reck.rec
    distances = []
    four_corners = [(R[0][0],R[0][1]),((R[0]+R[1]*s)[0],(R[0]+R[1]*s)[1]),((R[0]+R[2]*u)[0],(R[0]+R[2]*u)[1]),((R[0]+R[1]*s+R[2]*u)[0],((R[0]+R[1]*s+R[2]*u)[1]))]
    for corner in four_corners:
        distances.append(abs(Nm(vector(corner) - point)))
    return max(distances)

def closest_lattice_point(reck):
    "find lattice point with the smallest norm from the coordinates (bottom left corner) of the rectangle "
    rectangle = reck.rec
    results = []
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            results.append((abs(Nm(rectangle[0]-vector([x,y]))),vector((x,y))))
    winners = [results[0]]
    counter = 1
    while counter < 9:
        if results[counter][0] < winners[-1][0]:
            winners.append(results[counter])
        counter += 1
    return winners[-1][1]

def my_norm(reck):
    return max_distance(reck,closest_lattice_point(reck))

def partition_trapper(partition):
    new_partition = []
    for reck in partition:
        new_partition.append((reck,my_norm(reck)))
    return new_partition

def something_to_graph(partition,meet_matrix):
    counter = 0
    points = []
    length = len(partition)
    while counter < length:
        points.append((partition[counter].rec[3],haus(meet_matrix.matrix_from_rows(range(counter,length)).matrix_from_columns(range(counter,length)).eigenvalues())))
        counter += 1
    return points

def trapped(R,t):
    return bool((abs(Nm((R[0][0],R[0][1])))<t) and
               (abs(Nm(((R[0]+R[1]*s)[0],(R[0]+R[1]*s)[1])))<t) and
               (abs(Nm(((R[0]+R[2]*u)[0],(R[0]+R[2]*u)[1])))<t) and
               (abs(Nm(((R[0]+R[1]*s+R[2]*u)[0],(R[0]+R[1]*s+R[2]*u)[1])))<t))

# need: l_u>0, l_s<0
# what you begin with: A (matrix), Nm (form), s (stable eigenvector), u (unstable eigenvaector),
# M_orig (orig. 2-partition)
#
# still need drawing functions

l_u=1+sqrt(2); l_s=1-sqrt(2)

u=vector((1,sqrt(2)/2)); s=vector((1,-sqrt(2)/2))

def Nm(P):
    return abs(P[0]^2-2*P[1]^2)

#this appears to be the norm for the field

M_orig=[[vector((0,0)),1/2*sqrt(2),1/2], [vector((-1/2,sqrt(2)/4)),1/2,1/2*sqrt(2)]]

# need exact versions!
#l_u=max(A.eigenvalues())

#l_s=min(A.eigenvalues())

import itertools

T=(Matrix([s,u])^-1).transpose()

SR=parent(sqrt(2))

def rsimp(R):
    return [vector((SR(R[0][0]).expand(),SR(R[0][1]).expand())),SR(R[1]).expand(),SR(R[2]).expand()]

#expands and simplifies data?

def psimp(M):
    return [rsimp(R) for R in M]

#same?

def radd(v,R):
    return rsimp([vector(v)+R[0],R[1],R[2]])

#translates rectangle by given vector

def phi(R):
    return rsimp([A*(R[0]+R[1]*s), R[1]*(-l_s), R[2]*l_u])

def phi_inv(R):
    return rsimp([A^-1*(R[0]+R[1]*s), R[1]*l_u, R[2]*(-l_s)])

def phi_part(M):
    return [phi(R) for R in M]

def phi_inv_part(M):
    return [phi_inv(R) for R in M]

def rint0(A,B):
    b=True
    if bool((T*A[0])[0] <= (T*B[0])[0] < ((T*B[0])[0]+B[1]) <= ((T*A[0])[0]+A[1])):
       Ps=(T*B[0])[0]
       ds=B[1]
    elif bool((T*B[0])[0] <= (T*A[0])[0] < ((T*A[0])[0]+A[1]) <= ((T*B[0])[0]+B[1])):
       Ps=(T*A[0])[0]
       ds=A[1]
    else:
         b=False
    if bool((T*A[0])[1] <= (T*B[0])[1] < ((T*B[0])[1]+B[2]) <= ((T*A[0])[1]+A[2])):
       Pu=(T*B[0])[1]
       du=B[2]
    elif bool((T*B[0])[1] <= (T*A[0])[1] < ((T*A[0])[1]+A[2]) <= ((T*B[0])[1]+B[2])):
       Pu=(T*A[0])[1]
       du=A[2]
    else:
         b=False
    if b:
        return rsimp([T^-1*vector((Ps,Pu)), ds,du])
    else:
     return []

#????

def rint(R1,R2):
        R=[]
        xm1 = n(R1[0][0])
        xM1 = n((R1[0]+R1[1]*s+R1[2]*u)[0])
        ym1 = n((R1[0]+R1[1]*s)[1])
        yM1 = n((R1[0]+R1[2]*u)[1])
        xm2 = n(R2[0][0])
        xM2 = n((R2[0]+R2[1]*s+R2[2]*u)[0])
        ym2 = n((R2[0]+R2[1]*s)[1])
        yM2 = n((R2[0]+R2[2]*u)[1])
        for (i,j) in itertools.product([floor(xm1-xM2)+1..ceil(xM1-xm2)-1],[floor(ym1-yM2)+1..ceil(yM1-ym2)-1]):
            S=rint0(R1,radd((i,j),R2))
            if S != []:
                R.append(S)
        return R
#returns list of connected components

def part_int(X,Y):
        M = []
        for (R1,R2) in itertools.product(X,Y):
         R = rint(R1,R2)
            if R != []:
             M=M+rint(R1,R2)
        return M
#partition intersect

def meet_matrix(M):
    return Matrix([[int(rint(phi(R),S)!=[]) for S in M] for R in M])

#builds meet matrix
def trapped(R,t):
    return bool((abs(Nm((R[0][0],R[0][1])))<t) and (abs(Nm(((R[0]+R[1]*s)[0],(R[0]+R[1]*s)[1])))<t) and
        (abs(Nm(((R[0]+R[2]*u)[0],(R[0]+R[2]*u)[1])))<t) and
    (abs(Nm(((R[0]+R[1]*s+R[2]*u)[0],(R[0]+R[1]*s+R[2]*u)[1])))<t))

#returns true
def trapped_near(R,t):
    B=False
    for (i,j) in itertools.product([-1..1],[-1..1]):
        B=B or trapped(radd(vector((i,j)),R),t)
    return B


def weed_set(M,t):
    S=[]
    for R in M:
     if trapped_near(R,t):
    S.append(M.index(R))
    return(S)
#list of indices of trapped integrals (indexing probably starts at zero)

def weed_part(M,t):
    W =[]
    for R in M:
     if not(trapped_near(R,t)):
    W.append(R)
    return W

#actually does the weeding


# basic drawing stuff

def hyp(t,p=(0,0)):
    return plot(sqrt(((x-p[0])^2+t)/2)+p[1], [-2,2],color='black')+\
        plot(-sqrt(((x-p[0])^2+t)/2)+p[1], [-2,2],color='black')+\
        plot(sqrt(((x-p[0])^2-t)/2)+p[1], [sqrt(t)+p[0],2],color='black')+\
        plot(-sqrt(((x-p[0])^2-t)/2)+p[1], [sqrt(t)+p[0],2],color='black')+\
        plot(sqrt(((x-p[0])^2-t)/2)+p[1], [-2,-sqrt(t)+p[0]],color='black')+\
        plot(-sqrt(((x-p[0])^2-t)/2)+p[1], [-2,-sqrt(t)+p[0]],color='black')


def draw_nl(A):
    if len(A) == 3:
        return polygon([A[0],A[0]+A[1]*s,A[0]+A[1]*s+A[2]*u,A[0]+A[2]*u],alpha=.5,color='blue')
    else:return polygon([A[0],A[0]+A[1]*s,A[0]+A[1]*s+A[2]*u,A[0]+A[2]*u],alpha=.5,color=A[3])

def draw(A,label=''):
    if len(A) == 3:
        return polygon([A[0],A[0]+A[1]*s,A[0]+A[1]*s+A[2]*u,A[0]+A[2]*u],alpha=.5,color='blue',legend_label=label)
    else:
        return polygon([A[0],A[0]+A[1]*s,A[0]+A[1]*s+A[2]*u,A[0]+A[2]*u],alpha=.5,color=A[3],legend_label=label,legend_color=A[3])

def draw_boundary(R):
    return line([R[0],R[0]+R[1]*s,R[0]+R[1]*s+R[2]*u,R[0]+R[2]*u,R[0]],color='black')

def draw_nl_part(M):
    P = point((0,0))
    for m in M:
        P = P+draw_nl(m)+draw_boundary(m)
    return P


def draw_part(M):
    P = point((0,0))
    for m in M:
        P = P+draw(m,label=str(M.index(m)))+draw_boundary(m)
    return P


import random
rc = lambda: random.randint(0,255)


def color_part(M):
    return [m+['#%02X%02X%02X' % (rc(),rc(),rc())] for m in M]

#only works for colored partitions
def draw_trans_part(v,M):
    P = point((0,0))
    for m in M:
        P = P+draw([m[0]+v,m[1],m[2],m[3]])+draw_boundary([m[0]+v,m[1],m[2],m[3]])
    return P

#only works for colored partitions
def main_draw(M):
    P=point((0,0))
    return draw_part(M)+draw_trans_part(vector((-1,0)),M)+draw_trans_part(vector((-1,-1)),M)+draw_trans_part(vector((0,-1)),M)+draw_trans_part(vector((1,0)),M)

def rpt_phi_part(M, a):
    if a == 0:
        return M
    else:
        return rpt_phi_part(part_int(M,phi_part(M)),a - 1)

#returns phi^a


def rpt_phi_inv_part(M, a):
    if a == 0:
        return M
    else:
        return rpt_phi_inv_part(part_int(M, phi_inv_part(M)), a - 1)

#returns phi^-a

def das_repeaten(M, a):
    return part_int ( rpt_phi_inv_part(M, a), rpt_phi_part(M, a) )

# intersects phi^-a with phi^a

def itr_weed_part(M,t,d):
    Q=[]
    b=0
    while t-b*d>0:
        Q.append(weed_part(M,b*d))
        b += 1
    return Q

# (Partition, greatest t, incrementing value) Returns a list of the partition weeded out with various T's

def haus(eigenvalues):
    return 2*log(max(eigenvalues)/log(1+2.0^(1/2))