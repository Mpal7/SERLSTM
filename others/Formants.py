import numpy as np
import scipy.signal
import wave

class Class_get_fp(object):
    def __init__(self, NFRAME=640, NSHIFT=320, lpcOrder=32, FreqPoints=1024, max_num_formants=3):

        self.NFRAME = NFRAME  # 640 sr=16Khz 40mS  # 400 sr=16Khz 25mS
        self.NSHIFT = NSHIFT  # 320 sr=16Khz 20mS  # 160 sr=16khz 10mS
        self.lpcOrder = lpcOrder
        self.FreqPoints = FreqPoints  # need many number for precise analysis especially pitch detect
        self.window = np.hamming(self.NFRAME)  # Windows is Hamming
        self.preemph = 0.97  # pre-emphasis
        self.max_num_formants = max_num_formants  # maximum number of formant candidate to detect

    def get_fp(self,signal,fs):
        # read wave file
        nframes = signal.shape[0]
        self.df0 = (fs / 2.) / self.FreqPoints
        self.dt0 = 1.0 / fs

        # 16bit integer to float32
        fdata = signal.astype(np.float32)

        count = int(((nframes - (self.NFRAME - self.NSHIFT)) / self.NSHIFT))

        # prepare output
        spec_out = np.zeros([count, self.FreqPoints])
        fout = np.zeros([count, self.max_num_formants])
        fout_index = np.ones([count, self.max_num_formants]) * -1
        pout = np.zeros(count)
        pout_index = np.ones(count) * -1

        pos = 0  # position
        countr = 0

        for loop in range(count):

            ## copy to avoid original over-change
            frame = fdata[pos:pos + self.NFRAME].copy()

            ## pre-emphasis
            frame -= np.hstack((frame[0], frame[:-1])) * self.preemph
            ## do window
            windowed = self.window * frame
            ## get lpc coefficients
            a, e = lpc(windowed, self.lpcOrder)
            ## get lpc spectrum
            w, h = scipy.signal.freqz(np.sqrt(e), a, self.FreqPoints)  # from 0 to the Nyquist frequency
            lpcspec = np.abs(h)
            lpcspec[lpcspec < 1.0] = 1.0  # to avoid log(0) error
            loglpcspec = 20 * np.log10(lpcspec)
            spec_out[loop] = loglpcspec  # store to output
            ## get formant candidate
            f_result, i_result = self.formant_detect(loglpcspec, self.df0)

            if len(f_result) > self.max_num_formants:
                fout[loop] = f_result[0:self.max_num_formants]
                fout_index[loop] = i_result[0:self.max_num_formants]
            else:
                print(len(f_result))
                print(f_result)
                print(fout.shape)
                fout[loop] = f_result[0:len(f_result)]
                fout_index[loop] = i_result[0:len(f_result)]

            ## calcuate lpc residual error (= input source)
            r_err = residual_error(a, windowed)
            ## autocorrelation of lpc residual error (= input source)
            a_r_err = autocorr(r_err)
            a_f_result, a_i_result = self.pitch_detect(a_r_err, self.dt0)

            if len(a_f_result) > 0:  # if candidate exist,
                pout[loop] = a_f_result[0]
                pout_index[loop] = a_i_result[0]

                ## print output of candidates of [formants] pitch,  frequency[Hz]
                if countr == 0:
                    print('candidates of [formants] pitch,  frequency[Hz] ')
                print(fout[loop], pout[loop])

            # index count up
            countr += 1
            # next
            pos += self.NSHIFT

        return spec_out, fout, pout

    def formant_detect(self, input0, df0, f_min=250):
        is_find_first = False
        f_result = []
        i_result = []
        for i in range(1, len(input0) - 1):
            if f_min is not None and df0 * i <= f_min:
                continue
            if input0[i] > input0[i - 1] and input0[i] > input0[i + 1]:
                if not is_find_first:
                    f_result.append(df0 * i)
                    i_result.append(i)
                    is_find_first = True
                else:
                    f_result.append(df0 * i)
                    i_result.append(i)

        return f_result, i_result

    def pitch_detect(self, input0, dt0, ratio0=0.2, f_min=100, f_max=500):
        is_find_first = False
        f_result = []
        i_result = []
        v_result = []
        for i in range(1, len(input0) - 1):
            if np.abs(input0[i]) < np.abs(input0[0] * ratio0):
                continue
            fp = 1.0 / (dt0 * i)
            if f_max is not None and fp >= f_max:
                continue
            if f_min is not None and fp <= f_min:
                continue
            if input0[i] > input0[i - 1] and input0[i] > input0[i + 1]:
                if not is_find_first:
                    f_result.append(fp)
                    i_result.append(i)
                    v_result.append(input0[i])
                    is_find_first = True
                else:
                    f_result.append(fp)
                    i_result.append(i)
                    v_result.append(input0[i])
            elif input0[i] < input0[i - 1] and input0[i] < input0[i + 1]:
                if not is_find_first:
                    f_result.append(fp)
                    i_result.append(i)
                    v_result.append(input0[i])
                    is_find_first = True
                else:
                    f_result.append(fp)
                    i_result.append(i)
                    v_result.append(input0[i])

        if is_find_first:
            a = np.argmax(np.array(v_result))
            f_result2 = [f_result[np.argmax(np.array(v_result))]]
            i_result2 = [i_result[np.argmax(np.array(v_result))]]
        else:
            f_result2 = []
            i_result2 = []

        return f_result2, i_result2


def autocorr(x, nlags=None):
    N = len(x)
    if nlags == None: nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r


def LevinsonDurbin(r, lpcOrder):
    """
    Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める
    """
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]


# LPC係数を求める
#
#   入力：信号
#         LPCの次数
#
#   出力：LPC係数
#         差

def lpc(s, lpcOrder=32):
    r = autocorr(s, lpcOrder + 1)
    a, e = LevinsonDurbin(r, lpcOrder)
    return a, e


# LPC予測残差を計算する
#
#   入力：LPC係数
#         信号
#
#   出力：LPC予測残差

def residual_error(a, s):
    lpcOrder = len(a)
    r_error = s.copy()

    for i in range(lpcOrder, len(s)):
        for j in range(0, lpcOrder):
            r_error[i] += (a[j] * s[i - j - 1])
    r_error[0:lpcOrder - 1] = 0.0
    return r_error

def formants_extraction(signal,fs):
    fp0 = Class_get_fp()
    spec_out, fout, pout = fp0.get_fp(signal,fs)
    fout /= 255.0
