import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import struct
import sys
import io
import cv2

mpl.rcParams["lines.linewidth"] = 0.2


class TwoNodeSystem(object):
    def __init__(self, a1, a2, g1, g2, T1=1, T2=1, Transient_Iterations=None):
        """Generic Hopfield Neural Network with Two nodes and no
        self connections

        initial conditions:
            - a1, a2: internal neuron decay
            - g1, g2:
            - T1, T2:
        """
        # Default Value that will not iterate
        self.Transient_Iterations = 2
        if Transient_Iterations:
            self.Transient_Iterations = Transient_Iterations
        self.a1 = a1
        self.a2 = a2
        self.g1 = g1
        self.g2 = g2
        self.T1 = T1
        self.T2 = T2

        self.prev_bit = None
        self.next_bit = None

        self.x_vals = None
        self.y_vals = None

    def iterate(self, N, x0=None, x1=None, x2=None, y0=None, y1=None, y2=None):
        if x0 == None:
            x2 = self.x_vals[-1]
            x1 = self.x_vals[-2]
            x0 = self.x_vals[-3]

            y2 = self.y_vals[-1]
            y1 = self.y_vals[-2]
            y0 = self.y_vals[-3]

        self.x_vals = np.zeros(N + 1, dtype=np.longdouble)
        self.y_vals = np.zeros(N + 1, dtype=np.longdouble)

        self.x_vals[0] = x0
        self.x_vals[1] = x1
        self.x_vals[2] = x2

        self.y_vals[0] = y0
        self.y_vals[1] = y1
        self.y_vals[2] = y2

        # TODO Implement C Module to Run the Neural Netowork for Speed
        for n in range(3, N):
            self.x_vals[n + 1] = self.a1 * self.x_vals[n] + self.T1 * self.g1(
                self.y_vals[n - 2], dtype=np.longdouble
            )
            self.y_vals[n + 1] = self.a2 * self.y_vals[n] + self.T2 * self.g2(
                self.x_vals[n - 1], dtype=np.longdouble
            )
        return self.x_vals, self.y_vals

    def plot(self):
        plt.scatter(self.x_vals, self.y_vals)


def theta(t, Tau):
    """The Threshold function for our value of t"""
    if t < Tau:
        output = 0
    else:
        output = 1
    return output


# TODO optimize with C
def bit_binary_rep(t, d, e, i):
    p_of_two = [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        67108864,
        134217728,
        268435456,
        536870912,
        1073741824,
        2147483648,
        4294967296,
    ]

    bin_rep = 0
    for r in range(1, (2 ** i)):
        Tau = (e - d) * (r / (p_of_two[i])) + d
        bin_rep += ((-1) ** (r - 1)) * theta(t, Tau)
    return bin_rep


# TODO optimize with C
def Count(N, M, turn):
    def leftRotate(n, d):
        # In n<<d, last d bits are 0.
        # To put first 3 bits of n at
        # last, do bitwise or of n<<d
        # with n >>(INT_BITS - d)
        bin_array = np.array([np.binary_repr(N)])
        shifted_bin_array = np.roll(bin_array, -1 * d)
        left_shifted_int = int("0b" + "".join(x for x in shifted_bin_array), 2)
        return left_shifted_int

    # Function to right
    # rotate n by d bits
    def rightRotate(n, d):

        # In n>>d, first d bits are 0.
        # To put last 3 bits of at
        # first, do bitwise or of n>>d
        # with n <<(INT_BITS - d)

        bin_array = np.array([np.binary_repr(N)])
        shifted_bin_array = np.roll(bin_array, d)
        right_shifted_int = int("0b" + "".join(x for x in shifted_bin_array), 2)
        return right_shifted_int

    # Convert N into hexadecimal number and
    # remove the initial zeros in it
    if turn == "R":
        S = rightRotate(N, M)

    else:
        S = leftRotate(N, M)

    # Convert the rotated binary string
    # in decimal form, here 2 means in binary form.
    return S


def SegmentGenerator(Message, num_bytes=4):
    """
    Take a Byte String and Encode it into Chunks
    """

    while len(Message) != 0:
        if len(Message) < num_bytes:
            additions = num_bytes - len(Message)
            Message = Message + b" " * additions
        msg = Message[:num_bytes]
        Message = Message[num_bytes:]
        yield msg


def Compute_Binary_Sequence(rnn, first_time_flag=True):
    """
    Creates AJ and DJ
    """
    x_vals, y_vals = rnn.iterate(38)
    A_J = np.zeros(32, dtype=np.int8)
    D_J = np.zeros(5, dtype=np.int8)
    if rnn.prev_bit != 1:
        for k in range(1, 33):
            A_J[k - 1] = bit_binary_rep(x_vals[k - 1], -10, 0, i=4)
        q = 0
        for k in range(33, 38):
            D_J[q] = bit_binary_rep(x_vals[k - 1], -10, 0, i=4)
            # print(str(k)+".", D_J[q])
            q += 1
            rnn.prev_bit = x_vals[-1]
    else:
        # Choose y
        for k in range(1, 33):
            A_J[k - 1] = bit_binary_rep(y_vals[k - 1], -10, 0, i=4)
        # print(str(k)+".", A_J[k-1])
        q = 0
        for k in range(33, 38):
            D_J[q] = bit_binary_rep(y_vals[k - 1], -10, 0, i=4)
            # print(str(k)+".", D_J[q])
            q += 1
            rnn.prev_bit = y_vals[-1]

    AJ = "0b" + "".join([str(x) for x in A_J])
    DJ = "0b" + "".join([str(x) for x in D_J])

    return AJ, DJ, A_J, D_J


def decrypt(Message):
    b = -2
    my_rnn_pos = TwoNodeSystem(
        a1=1 / 4, a2=3 / 4, g1=np.sin, g2=np.tanh, T1=1, T2=b
    )
    x_N, y_N = my_rnn_pos.iterate(
        N=10001, x0=0.01, x1=0.01, x2=0.01, y0=0.01, y1=0.01, y2=0.01
    )
    x0 = x_N[-1:]

    # STEP-2: Break the Message into l=4 subsequences
    getChunk = SegmentGenerator(Message, num_bytes=4)

    final_message = b""
    print("Decrypting")
    for i, chunk in enumerate(getChunk):
        print("iteration {curr}".format(curr=i), end="\r")
        Cj = chunk
        # print("\n",chunk, "LENGTH:", len(chunk))

        # STEP-3: Compute the Binary Sequences Supplied by the Fourth bits

        # print("PJ:", PJ)
        # print("LEN:", len(PJ))

        AJ, DJ, A_J, D_J = Compute_Binary_Sequence(my_rnn_pos)
        ai = int(AJ, 2)
        Ci = int.from_bytes(Cj, "little")
        a1_prime = Count(ai, int(DJ, 2), "R")
        P1 = a1_prime ^ Ci
        plaintext = Count(P1, int(DJ, 2), "R")

        final_message += int.to_bytes(plaintext, 4, "little")
    decrypted_message = final_message
    return decrypted_message


def encrypt(Message):
    b = -2
    my_rnn_pos = TwoNodeSystem(
        a1=1 / 4, a2=3 / 4, g1=np.sin, g2=np.tanh, T1=1, T2=b
    )
    x_N, y_N = my_rnn_pos.iterate(
        N=10001, x0=0.01, x1=0.01, x2=0.01, y0=0.01, y1=0.01, y2=0.01
    )
    x0 = x_N[-1:]

    # STEP-2: Break the Message into l=4 subsequences
    getChunk = SegmentGenerator(Message, num_bytes=4)

    final_message = b""
    print("Encrypting ...")
    for i, chunk in enumerate(getChunk):
        print("iteration {curr}".format(curr=i), end="\r")
        PJ = chunk
        # print("CHUNK:", chunk ,"LEN:", len(chunk))
        AJ, DJ, A_J, D_J = Compute_Binary_Sequence(my_rnn_pos)
        a1 = int(AJ, 2)
        p1 = int.from_bytes(PJ, "little")

        # print("before")
        # print("Pj:", bin(p1))
        # print("Aj:", bin(a1))

        p1_prime = Count(p1, int(DJ, 2), "L")
        a1_prime = Count(a1, int(DJ, 2), "R")
        # print("after")
        # print("Pj' Integer:", bin(p1_prime))
        # print("Aj' Integer:", bin(a1_prime))

        cypher = p1_prime ^ a1_prime
        # print("CYPHER", cypher)

        # x_N, y_N = my_rnn_pos.iterate(N=38)
        final_message += int.to_bytes(cypher, 4, "little")
    encrypted_message = final_message
    return encrypted_message


def encrypt_image(PATH_TO_IMAGE, progressbar=None):
    img = cv2.imread(PATH_TO_IMAGE, 1)
    # cv2.imshow('',img)
    shape = img.shape
    print(shape)
    if progressbar:
        progressbar.set_fraction(0.25)
    img2 = img.flatten()
    upper_bound = img2.shape[0]
    print(upper_bound)
    byte_img = img2.tobytes()
    if progressbar:
        progressbar.set_fraction(0.70)
    enc_img = encrypt(byte_img)
    nparr = np.frombuffer(enc_img, np.uint8)
    img_np = nparr[:upper_bound].reshape(
        *shape
    )  # cv2.IMREAD_COLOR in OpenCV 3.1
    # cv2.imshow("encrypted image", img_np)
    # cv2.waitKey(0)
    if progressbar:
        progressbar.set_fraction(0.99)
    cv2.imwrite(PATH_TO_IMAGE, img_np)
    return img_np


def decrypt_image(PATH_TO_IMAGE, progressbar=None):
    img = cv2.imread(PATH_TO_IMAGE, 1)
    # cv2.imshow('',img)
    shape = img.shape
    print(shape)
    if progressbar:
        progressbar.set_fraction(0.25)
    img2 = img.flatten()
    upper_bound = img2.shape[0]
    byte_img = img2.tobytes()
    if progressbar:
        progressbar.set_fraction(0.70)
    enc_img = decrypt(byte_img)
    nparr = np.frombuffer(enc_img, np.uint8)
    img_np = nparr[:upper_bound].reshape(
        *shape
    )  # cv2.IMREAD_COLOR in OpenCV 3.1
    # cv2.imshow("decrypted image", img_np)
    # cv2.waitKey(0)
    if progressbar:
        progressbar.set_fraction(0.99)
    cv2.imwrite(PATH_TO_IMAGE, img_np)
    return img_np
