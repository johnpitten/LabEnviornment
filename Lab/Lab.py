import time
import skrf as rf
import numpy as np
from typing import Sequence
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

from bcqthub.controllers import HEMTController #This is Jorge's bcqt-hub-revamp
from bcqt_hub.drivers.misc.MiniCircuits.MC_VarAttenuator import MC_VarAttenuator #This is the actual bcqt-hub
from CryoSwitchController import Cryoswitch

#TODO: fix connection issue when there are no active measurements -- bug in PNA.create_channel(1, "Channel 1")
#TODO: add setter for arbitrary freq points

#TODO: add programmable attenuators for power setting
class EnhancedPNA(PNA):
    _models = {
        "default": {"nports": 2, "unsupported": []},
        "E8362C": {"nports": 2, "unsupported": ["nports", "freq_step", "fast_sweep"]},
        "N5227B": {"nports": 4, "unsupported": []},
    }

    class Channel(PNA.Channel):
        def __init__(self, parent, cnum: int, cname: str):
            super().__init__(parent, cnum, cname)



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
            validator = IntValidator(-90, 0),
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

        if self.parent.ext_attenuation:
            def setPower(self, P:float):
                if P >= -90:
                    self.parent.SetAttn(0)
                    self.power_level = P
                elif P >= -120 and P < -90:
                    self.parent.SetAttn(-P-90)
                    self.power_level = -90
                    print(f'VNA Power: {P} dBm, external attenuation: {-P-90} dB')
                elif P < -120:
                    raise ValueError('Cannot reach power levels below -120 dBm')

            def getPower(self):
                P = self.power_level
                a = self.parent.GetAttn()
                return P-a


        '''
        Set/get the number of traces of selected channel: CALCulate:PARameter:COUNt
        Delete Trace: DISPlay:WINDow:TRACe:DELete
        Add Trace: DISPlay:WINDow:TRACe[:STATe]
        New Trace: DISPlay:WINDow:TRACe[:STATe]
        Select Trace: DISPlay:WINDow:TRACe:SELect
        '''

    def __init__(self, address: str, backend: str = "@py", ext_attenuators = False) -> None:
        #How can the attenuators variable be passed to the Channel subclass' __init__ function?
        super(PNA, self).__init__(address, backend)
        #this references the initialization of skrf's VNA class since we need to modify the PNA __init__ function
        #to fix connection issues when there is no active measurement channel

        self._resource.read_termination = "\n"
        self._resource.write_termination = "\n"

        #this block is modified from the skrf PNA class so we can connect despite
        #there being no active measurement channels

        #WARNING: the SCPI command CALCulate<cnum>:PARameter:COUNt returns 1 when there are 0 active measurements
        self.create_channel(1, "Channel 1")
        #query list of measurements
        ms = self.ch1.measurements
        print(f'ms: {ms}')
        if ms == []:
            print('WARNING: No currently active measurement channels.')
            #may need to additionally check for existence of window before creating a new one
            self.write('DISP:WIND ON')
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
        if attenuators:
            print('Connecting to programmable attenuators')

            self.att1 = MC_VarAttenuator(device_address = "192.168.0.113")
            self.att2 = MC_VarAttenuator(device_address = "192.168.0.114")

            #these have methods att.Set_Attenuation(atten: float), and att.Get_Attenuation()
            #must define pna.setAttn(), pna.getAttn(), pna.Channel.setPower()
            def getAttn(self):
                atten1 = float(att1.Get_Attenuation()[1])
                atten2 = float(att2.Get_Attenuation()[1])

                if atten1 != atten2:
                    raise ValueError('dissimilar attenuation values')
                else:
                    return atten1

            def setAttn(a: float):
                if a < 0 or a > 30:
                    raise ValueError('attenuation must be between 0 and 30 dB')
                else:
                    att1.Set_Attenuation(a)
                    att2.Set_Attenuation(a)

            #set attenuation to 0 upon initialization
            self.SetAttn(0)
            self.ext_attenuation = ext_attenuators

            '''
            import clr  # pythonnet (do not pip install clr, pip install pythonnet instead)
            clr.AddReference('mcl_RUDAT_NET45')  # Reference the DLL
            #after downloading the .ddl file from minicircuits the .zip file has to be unblocked
            #if this option does not appear in general properties of the .zip file then you must
            #do so in Windows Powershell (run as Administrator)
            #the command is: unblock-file -path "string to path"
            # the .zip extention shuld be included

            from mcl_RUDAT_NET45 import USB_RUDAT
            #need to think about how to incorporate attenuators
            self.att1 = USB_RUDAT()
            self.att2 = USB_RUDAT()

            Status1 = self.att1.Connect(SN = str(attenuators['Port 1']))
            Status2 = self.att2.Connect(SN = str(attenuators['Port 2']))
            #TODO: actually check the status of these connections
            #set attenuation to 0 on both attenuators
            #define coordinated set & get attenuation functions

        else:
            print('No external attenuators')
            '''



#change inputs so I don't have to deal with cfg dictionary
class LabSwitch(Cryoswitch):
    def __init__(self, switch_debug=False, COM_port='', switch_IP=None, SN=None, override_abspath=False,
                 configs = dict(), HEMTctrl_debug=False, **kwargs):
        #May want to have self.switch = Cryoswitch(...) and self.HEMTctrl = HEMTController(...) instead
        Cryoswitch.__init__(self, debug=switch_debug, COM_port=COM_port, IP=switch_IP, SN=SN,
                            override_abspath=override_abspath)
        self.ctrl = HEMTController(configs = configs, debug=HEMTctrl_debug, **kwargs)
        self.ctrl.gate_value = 1.1
        self.ctrl.drain_value = 0.7
        self.ctrl.step = 0.05
        self.ctrl.delay = 0.2
        self.start()#Cryoswitch method
        self.set_output_voltage(5.5)
        self.devices = dict()
        #keep track of active switch channels?

    def safeConnect(self, channel: int | str, safe_mode = True):
        #check channel for type (int or string) if string then pass to self.devices dict to get the channel number
        if type(channel) == str:
            channel_number = devices[channel]
        elif type(channel) == int:
            channel_number = channel

        #Ramp Down the HEMT power supply
        self.ctrl.turn_off(step=self.ctrl.step, delay=self.ctrl.delay)
        #check that the HEMTs are off
        while safe_mode:
            user_check = input('Check that the HEMTs are powered off, enter y/n:')
            if user_check == 'n':
                raise ValueError('HEMTs did not power off as expected')
            elif user_check != 'y':
                print('enter either \'y\' or \'n\'')
            elif user_check == 'y':
                break
        self.disconnect_all(port='A')
        time.sleep(1)
        self.disconnect_all(port='B')
        time.sleep(1)
        self.connect(port='A', contact=channel_number)
        time.sleep(1)
        self.connect(port='B', contact=channel_number)
        self.ctrl.turn_on(gate_stop=self.ctrl.gate_value, drain_stop=self.ctrl.drain_value,
                           step=self.ctrl.step, delay=self.ctrl.delay)
        display_string = f'Cryoswitch is now on channel {channel_number}'
        if type(channel) == str:
            display_string = display_string + f' (DUT: {channel})'
        print(display_string)
