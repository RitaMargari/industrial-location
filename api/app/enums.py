from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class IndustryEnums(str, Enum):
    MINING_OIL_GAS = 'mining_oil_gas'
    MINING_COAL_ORES = 'mining_coal_ores'
    MECH_ENGINEERING = 'mech_engineering'
    SHIPBUILDING = 'shipbuilding'
    AIRCRAFT_ENGINEERING = 'aircraft_engineering'
    NON_FERROUS_METALLURGY = 'non_ferrous_metallurgy'
    FERROUS_METALLURGY = 'ferrous_metallurgy'
    CHEMICAL = 'chemical'
    PHARMA = 'pharma'
    ELECTRONICS = 'electronics'


class WorkForce(str, Enum):
    ALL = 'all'
    SPECIALISTS = 'specialists'
    GRADUATES = 'graduates'

class Transportation(str, Enum):
    PRIVATE_CAR = 'private_car'
    PUBLIC_TRANSPORT = 'public_transport',

class Cities(str, Enum):
    SAINT_PETERSBURG = 'saint-petersburg'
    TOMSK = 'tomsk'
    PERM = 'perm'
    SHAKHTY = 'shakhty'