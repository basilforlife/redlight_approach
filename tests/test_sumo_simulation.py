from redlight_approach.sumo_simulation import SumoSimulation


class TestSumoSimulation:
    def test_type(self, sumo_sim):
        assert isinstance(sumo_sim, SumoSimulation)

    def test_sumo_command(self, sumo_sim):
        assert sumo_sim.sumo_command == [
            "/usr/local/bin/sumo",
            "--configuration-file",
            "sumo/two_roads/f.sumocfg",
            "--step-length",
            "0.5",
            "--route-files",
            "sumo/two_roads/f_modified.rou.xml",
            "--fcd-output",
            "sumo/two_roads/fcd.xml",
        ]
