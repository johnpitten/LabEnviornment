import time
import skrf as rf
import numpy as np
from typing import Sequence
from types import MethodType
import itertools
import sys
from skrf.vi.vna import ValuesFormat
from skrf.vi.vna.keysight import PNA
from skrf.vi.validators import (
    BooleanValidator,
    DelimitedStrValidator,
    EnumValidator,
    FloatValidator,
    FreqValidator,
    IntValidator,
)

from bcqthubrevamp.controllers.HEMTController import HEMTController #This is Jorge's bcqt-hub-revamp
from bcqthub.drivers.misc.MiniCircuits.MC_VarAttenuator import MC_VarAttenuator #This is the actual bcqt-hub
from CryoSwitchController import Cryoswitch
from InstrumentAddresses import attenuator1_IP, attenuator2_IP


#TODO: add setter for arbitrary freq points
class EnhancedPNA(PNA):
    _models = {
        "default": {"nports": 2, "unsupported": []},
        "E8362C": {"nports": 2, "unsupported": ["nports", "freq_step", "fast_sweep"]},
        "N5227B": {"nports": 4, "unsupported": []},
        'N5231B': {"nports": 2, "unsupported": []}, #not added by scikit-rf

    }

    class Channel(PNA.Channel):
        def __init__(self, parent, cnum: int, cname: str):
            super().__init__(parent, cnum, cname)

            if self.parent.ext_attn:

                def setPower(self, P: float):
                    if P >= -90:
                        self.parent.setAttn(0)
                        self.power_level = P
                    elif P >= -120 and P < -90:
                        self.parent.setAttn(-P - 90)
                        self.power_level = -90
                        print(f'VNA Power: {P} dBm, external attenuation: {-P - 90} dB')
                    elif P < -120:
                        raise ValueError('Cannot reach power levels below -120 dBm')

                def getPower(self):
                    P = self.power_level
                    a = self.parent.getAttn()
                    return P - a

                #Bind functions to Channel object
                self.setPower = MethodType(setPower, self)
                self.getPower = MethodType(getPower, self)



        #Returns the sum of weights for each associated trace:
        #tr1: 2
        #tr2: 4
        #tr3: 8
        #tr4: 16
        #so when traces 1 and 2 have finished averaging, this returns 2+4 = 6
        #or when traces 3 and 4 have finished averaging this returns 8+16 = 24
        #TODO: query active trace numbers
        avg_status = PNA.command(
            get_cmd= "STAT:OPER:AVER1:COND?",
            set_cmd= None,
            doc="""Whether averaging has been completed. Returns the sum  2**n over each 
            trace n which has completed averaging.""",
            validator=IntValidator(),
        )

        power_level = PNA.command(
            get_cmd = "SOUR<self:cnum>:POW?",
            set_cmd = "SOUR<self:cnum>:POW <arg>",
            doc="""RF Power level in dBm""",
            validator = FloatValidator(-90, 0),
        )

        rfpower = PNA.command(
            get_cmd = "OUTP:STAT?",
            set_cmd = "OUTP:STAT <arg>",
            doc = """Toggle the RF Power On/Off""",
            validator = BooleanValidator(),
        )

        #Query names of active measurements: CALCulate<cnum>:PARameter:CATalog:EXTended? <enum>
        #Query number of active measurements: CALCulate<cnum>:PARameter:COUNt <value>

        def delete_all_measurements(self):
            meas = self.measurement_names
            for n in range(len(meas)):
                self.delete_measurement(meas[n])
                time.sleep(0.1)

        def select_trace(self, name: str):
            self.write(f"CALC{self.cnum}:PAR:SEL \'{name}\'")


        #modified version of get_snp_network which assumes that the measurement has been completed
        def get_s2p_network(
                self,
                ports: Sequence | None = None,
        ) -> rf.Network:
            if ports is None:
                ports = list(range(1, self.parent.nports + 1))

            orig_query_fmt = self.parent.query_format
            self.parent.query_format = ValuesFormat.BINARY_64
            self.parent.active_channel = self
            orig_snp_fmt = self.query("MMEM:STOR:TRAC:FORM:SNP?")
            self.write("MMEM:STOR:TRACE:FORM:SNP RI")  # Expect Real/Imaginary data

            msmnt_params = [f"S{a}{b}" for a, b in itertools.product(ports, repeat=2)]

            names = ['S11', 'S21', 'S12', 'S22']
            # Make sure the ports specified are driven

            # for param in msmnt_params:
            #     # Not all models support CALC:PAR:TAG:NEXT
            #     name = f"CH{self.cnum}_SKRF_{param}"
            #     names.append(name)
            #     self.create_measurement(name, param)

            #self.sweep()

            port_str = ",".join(str(port) for port in ports)
            raw = self.query_values(
                f"CALC{self.cnum}:DATA:SNP:PORTS? '{port_str}'", container=np.array
            )
            self.parent.wait_for_complete()

            # The data is sent back as:
            # [
            #   [frequency points],
            #   [s11.real],
            #   [s11.imag],
            #   [s12.real],
            #   [s12.imag],
            # ...
            # ]
            # but flattened. So we recreate the above shape from the flattened data
            npoints = self.npoints
            nrows = len(raw) // npoints
            nports = len(ports)
            data = raw.reshape((nrows, -1))[1:]

            ntwk = rf.Network()
            ntwk.frequency = self.frequency
            ntwk.s = np.empty(
                shape=(len(ntwk.frequency), nports, nports), dtype=complex
            )
            real_rows = data[::2]
            imag_rows = data[1::2]
            for n in range(nports):
                for m in range(nports):
                    i = n * nports + m
                    ntwk.s[:, n, m] = real_rows[i] + 1j * imag_rows[i]

            self.parent.query_format = orig_query_fmt
            self.write(f"MMEM:STOR:TRACE:FORM:SNP {orig_snp_fmt}")

            return ntwk




        '''
        Set/get the number of traces of selected channel: CALCulate:PARameter:COUNt
        Delete Trace: DISPlay:WINDow:TRACe:DELete
        Add Trace: DISPlay:WINDow:TRACe[:STATe]
        New Trace: DISPlay:WINDow:TRACe[:STATe]
        Select Trace: DISPlay:WINDow:TRACe:SELect
        '''

    def __init__(self, address: str, backend: str = "@py", ext_attenuators = False) -> None:
        self.ext_attn = ext_attenuators
        super(PNA, self).__init__(address, backend)
        #this references the initialization of skrf's VNA class since we need to modify the PNA __init__ function
        #to fix connection issues when there is no active measurement channel

        self._resource.read_termination = "\n"
        self._resource.write_termination = "\n"

        #this block is modified from the skrf PNA class so we can connect despite
        #there being no active measurement channels

        #WARNING: the SCPI command CALCulate<cnum>:PARameter:COUNt returns 1 when there are 0 active measurements
        #To the best of my knowledge, self.create_channel does not send and SCPI commands to the vna,
        # but self.active_channel does
        self.create_channel(1, "Channel 1")
        #query list of measurements
        ms = self.ch1.measurements
        if ms == []:
            print('WARNING: No currently active measurement channels.')
            self.write('DISP:WIND ON')
            self.active_channel = self.ch1
        else:
            self.active_channel = self.ch1

        self.model = self.id.split(",")[1]
        if self.model not in self._models:
            print(
                f"WARNING: This model ({self.model}) has not been tested with "
                "scikit-rf. By default, all features are turned on but older "
                "instruments might be missing SCPI support for some commands "
                "which will cause errors. Consider submitting an issue on GitHub to "
                "help testing and adding support.",
                file=sys.stderr,
            )
        self.query_format = ValuesFormat.BINARY_64

        #connect to external attenuators if Serial Numbers given
        #format is {'Port 1': '12345', 'Port 2': '67890'}
        if ext_attenuators:
            print('Connecting to programmable attenuators')


            self.att1 = MC_VarAttenuator(device_address = attenuator1_IP, debug = False)
            self.att2 = MC_VarAttenuator(device_address = attenuator2_IP, debug = False)


            def getAttn(self):
                atten1 = float(self.att1.Get_Attenuation()[1])
                atten2 = float(self.att2.Get_Attenuation()[1])

                if atten1 != atten2:
                    raise ValueError('dissimilar attenuation values')
                else:
                    return atten1

            def setAttn(self, a: float):
                if a < 0 or a > 30:
                    raise ValueError('attenuation must be between 0 and 30 dB')
                else:
                    self.att1.Set_Attenuation(a)
                    self.att2.Set_Attenuation(a)
            #Bind these functions to the pna object i.e. self
            #I'm not actually sure why this works
            self.getAttn = MethodType(getAttn, self)
            self.setAttn = MethodType(setAttn, self)
            #set attenuation to 0 upon initialization
            self.setAttn(0)

        else:
            print('No external attenuators')
            self.ext_attn = False

    #active_channel had to be overwritten to handle initialization without active measurements
    @property
    def active_channel(self) -> Channel | None:
        num = int(self.query("SYST:ACT:CHAN?"))
        return getattr(self, f"ch{num}", None)


    @active_channel.setter
    def active_channel(self, ch: Channel) -> None:
        if self.active_channel is None:
            #create a dummy measurement
            name = 'DUMMY_S11'
            parameter = 'S11'
            #hard-coded the create_measurement method without reference to ch.cnum, because the measurement
            #needs to be created before ch.cnum exists
            #SCPI supports this, but self.create_measurement does not
            self.write(f"CALC:PAR:EXT '{name}',{parameter}")
            # Not all instruments support DISP:WIND:TRAC:NEXT
            traces = self.query("DISP:WIND:CAT?").replace('"', "")
            traces = [int(tr) for tr in traces.split(",")] if traces != "EMPTY" else [0]
            next_tr = traces[-1] + 1
            self.write(f"DISP:WIND:TRAC{next_tr}:FEED '{name}'")

            msmnt = ch.measurement_numbers[0]
            print(msmnt)
            self.write(f"CALC{ch.cnum}:PAR:MNUM {msmnt}") #this sets the active measurement using the trace number
            #no command to activate new channel
            self.delete_all_measurements()
            return

        elif self.active_channel.cnum is ch.cnum:
            return

        msmnt = ch.measurement_numbers[0]
        self.write(f"CALC{ch.cnum}:PAR:MNUM {msmnt}")






#TODO: keep track of active switch channels after putting switch into known initial state (safely)
#TODO: suppress tqdm progress bar
class LabSwitch(Cryoswitch):
    def __init__(self, switch_debug=False, COM_port='', switch_IP=None, SN=None, override_abspath=False,
                 HEMTctrl_address = '', suppress_logs = True, **kwargs):
        #May want to have self.switch = Cryoswitch(...) and self.HEMTctrl = HEMTController(...) instead
        Cryoswitch.__init__(self, debug=switch_debug, COM_port=COM_port, IP=switch_IP, SN=SN,
                            override_abspath=override_abspath)
        configs = {'address': HEMTctrl_address}
        self.ctrl = HEMTController(configs = configs, suppress_logs = suppress_logs, **kwargs)

        self.ctrl.gate_setpoint = 1.1
        self.ctrl.drain_setpoint = 0.7
        self.ctrl.step = 0.05
        self.ctrl.delay = 0.01

        # np.arange excludes the endpoint
        self.ctrl.gate_vs = np.arange(0, self.ctrl.gate_setpoint+0.5*self.ctrl.step, self.ctrl.step)
        self.ctrl.drain_vs = np.arange(0, self.ctrl.drain_setpoint+0.5*self.ctrl.step, self.ctrl.step)

        self.start()#Cryoswitch method
        self.set_output_voltage(5.5)
        self.devices = dict()

        def getVoltage(self, channel):
            return self.psu.get_channel_voltage(channel)
        self.ctrl.getVoltage = MethodType(getVoltage, self.ctrl)




    def safeConnect(self, channel: int | str, safe_mode = False):
        #check channel argument for type (int or string) if string then pass to self.devices dict to get the channel number
        if type(channel) == str:
            channel_number = self.devices[channel]
        elif type(channel) == int and 1 <= channel <= 6:
            channel_number = channel
        else:
            raise ValueError('unexpected input for switch connection')

        #check HEMT output status, ramp down if on
        gate_status = self.ctrl.psu.get_output(self.ctrl.gate_channel)
        drain_status = self.ctrl.psu.get_output(self.ctrl.drain_channel)
        if gate_status == True and drain_status == True:
            print('HEMTs are on, ramping voltage biases down')
            self.ctrl.turn_off(step=self.ctrl.step, delay=self.ctrl.delay)
        elif gate_status == False and drain_status == False:
            print('HEMTs are already off')
        else:
            raise ValueError('Unexpected HEMT bias')


        #user check that the HEMTs are off
        while safe_mode:
            user_check = input('Check that the HEMTs are powered off, enter y/n:')
            if user_check == 'n':
                raise ValueError('HEMTs did not power off as expected')
            elif user_check != 'y':
                print('enter either \'y\' or \'n\'')
            elif user_check == 'y':
                break


        #operate the switch
        self.disconnect_all(port='A')
        time.sleep(1)
        self.disconnect_all(port='B')
        time.sleep(1)
        self.connect(port='A', contact = channel_number)
        time.sleep(1)
        self.connect(port='B', contact = channel_number)


        #Ramp HEMT bias back on
        self.ctrl.turn_on(gate_voltages=self.ctrl.gate_vs, drain_voltages=self.ctrl.drain_vs,
                            delay=self.ctrl.delay)

        #print current switch channel
        display_string = f'Cryoswitch is now on channel {channel_number}'
        if type(channel) == str:
            display_string = display_string + f' (DUT: {channel})'
        print(display_string)


