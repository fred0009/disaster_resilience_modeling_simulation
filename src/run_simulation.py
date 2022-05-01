import simpy
import random
import time
import networkx as nx
from operator import itemgetter

class Monitoring:
    def __init__(self, responding_hospitals, env):
        self.total_untreated_victims_series = []
        self.total_treated_victims_series = []
        self.critical_functionality_series = []
        env.process( self.start_monitoring_system_performance(responding_hospitals, env) )

    def start_monitoring_system_performance(self, responding_hospitals, env):
        while True:
            total_treated_victims = sum([x.treated_victims for x in responding_hospitals])
            total_untreated_victims = sum([x.untreated_victims + x.untreated_victims_on_the_way
                                           for x in responding_hospitals])
            try:
                critical_functionality = total_treated_victims/(total_treated_victims+total_untreated_victims)
            except ZeroDivisionError:
                critical_functionality = 1

            self.total_untreated_victims_series.append(total_untreated_victims)
            self.total_treated_victims_series.append(total_treated_victims)
            self.critical_functionality_series.append( critical_functionality )
            yield env.timeout(30)



def start_simulation(disaster_list, responding_hospitals, list_of_hospitals, health_offices, communication_duration,
                     central_hospitals, scenario, env):
    disaster_event(disaster_list, responding_hospitals, health_offices, communication_duration, central_hospitals,
                   scenario, env)
    monitoring = Monitoring(responding_hospitals, env)
    return monitoring

def disaster_event(disaster_list, responding_hospitals, health_offices, communication_duration, central_hospitals, scenario, env):
    for disaster in disaster_list:
        numbers_of_shared_hospitals = len(disaster.hospital_and_driving_time)
        victims_per_hospital = round( disaster.victims/numbers_of_shared_hospitals )
        for hospital, driving_time in disaster.hospital_and_driving_time.items():
            env.process( victims_coming_to_hospitals(hospital, driving_time, victims_per_hospital, health_offices,
                                                     communication_duration, central_hospitals, scenario, env) )


def help_hospital(health_office, hospital, com_duration, central_hospitals, env):
    for neighbor_hospital, driving_time in hospital.neighbors:
        if neighbor_hospital in health_office.hospitals_in_administration:
            if neighbor_hospital.ID in central_hospitals:
                communication_duration = 0
            else:
                communication_duration = com_duration

            untreated_victims = hospital.untreated_victims
            avail_beds = neighbor_hospital.available_beds

            # print("{} send victims to {} as asked by {} at {}".format(hospital.ID,neighbor_hospital.ID, health_office.ID, env.now))
            # print('currently {} has {} victims and {} beds, hospital {} has {} beds '.format(hospital.ID, untreated_victims, hospital.available_beds, neighbor_hospital.ID, neighbor_hospital.available_beds))

            if avail_beds > 0:
                if avail_beds >= untreated_victims:
                    neighbor_hospital.available_beds -= untreated_victims
                    hospital.untreated_victims = 0
                    hospital.untreated_victims_on_the_way += untreated_victims
                    with health_office.communication_channel.request() as req:
                        yield req
                        yield env.timeout(communication_duration)
                        env.process( contact_hospital_for_referral(hospital, neighbor_hospital, driving_time,
                                                                untreated_victims, env) )
                    break
                else:
                    neighbor_hospital.available_beds = 0
                    hospital.untreated_victims -= avail_beds
                    hospital.untreated_victims_on_the_way += avail_beds
                    with health_office.communication_channel.request() as req:
                        yield req
                        yield env.timeout(communication_duration)
                        env.process( contact_hospital_for_referral(hospital, neighbor_hospital, driving_time,
                                                               avail_beds, env))
    # if DHO does not have enough resources, it will contact PHO
    if hospital.untreated_victims > 0:
        if health_office.ID != 'PHO':
            with health_office.communication_channel.request() as req_DHO:
                yield req_DHO
                with health_office.PHO.communication_channel.request() as req_PHO:
                    yield req_PHO
                    # print("DHO from {} ask PHO's help at {}".format(health_office.ID, env.now))
                    yield env.timeout(com_duration)

                    env.process( help_hospital(health_office.PHO, hospital, com_duration, central_hospitals, env) )
        else:
            # print("No more available beds")
            pass



def contact_hospital_for_referral(hospital, neighbor_hospital, driving_time, transferred_victims, env):
    yield env.timeout(driving_time)
    hospital.untreated_victims_on_the_way -= transferred_victims
    hospital.treated_victims += transferred_victims


def victims_coming_to_hospitals(hospital, travel_time, victims, health_offices, communication_duration, central_hospitals,
                                scenario, env):
    yield env.timeout(travel_time)
    diff = hospital.available_beds - victims
    if diff >= 0:
        hospital.available_beds -= victims
        hospital.treated_victims += victims
    else:
        hospital.available_beds = 0
        hospital.treated_victims += victims+diff
        hospital.untreated_victims += abs(diff)
        if scenario == 'PHO':
            with health_offices.communication_channel.request() as req:
                yield req
                yield env.timeout(communication_duration)
                env.process( help_hospital(health_offices, hospital, communication_duration, central_hospitals, env) )
        elif scenario == 'DHO':
            with hospital.health_office.communication_channel.request() as req:
                yield req
                yield env.timeout(communication_duration)
                env.process( help_hospital(hospital.health_office, hospital, communication_duration,
                                           central_hospitals, env) )





if __name__ == "__main__":
    env = simpy.Environment()