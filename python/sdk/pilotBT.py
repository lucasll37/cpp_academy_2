"""
PilotBT — Modelo de decisão para o agente AsaPilotBT.

Implemente os métodos `predict` e `get_schema`.
Todo o resto (entradas, warmup, load, unload, input de exemplo) já está
resolvido pela classe base `PilotBTModel`.
"""

from __future__ import annotations
from typing import Any
from miia_model import PilotBTModel, ModelSchema

import json
import numpy as np
import numpy.typing as npt


class PilotBT(PilotBTModel):

    # -------------------------------------------------------------------------
    # Lógica de decisão
    # -------------------------------------------------------------------------

    def predict(self, inputs: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
        """Lógica de decisão do piloto autônomo.

        Parâmetros
        ----------
        inputs : dict[str, Any]
            Estado completo do agente.

        Retorna
        -------
        dict[str, np.ndarray]
            Um tensor ``[1, 1]`` float32 por chave declarada no método `get_schema`.
            O servidor extrai o valor escalar de cada tensor e o envia ao simulador.
        """

        # ── Grupos principais ─────────────────────────────────────────────────
        sim   = inputs.get("sim", {})           # fase e tempo de simulação
        own   = inputs.get("ownship", {})       # estado da própria aeronave
        area  = inputs.get("game_area", {})     # limites da área de jogo
        terr  = inputs.get("territory", {})     # limites do território
        faor  = inputs.get("faor", {})          # área de responsabilidade
        inv   = inputs.get("inventory", {})     # combustível e armamento
        shrd  = inputs.get("shared", {})        # formação, CAP e atribuições
        nav   = inputs.get("nav", {})           # waypoint ativo e navegação
        tdata = inputs.get("terrain_data", {})  # elevação do terreno à frente
        radar = inputs.get("radar", {})         # modo atual do radar
        trks  = inputs.get("air_tracks", [])    # tracks radar/datalink/RWR
        gatk  = inputs.get("gnd_atk", {})       # dados de ataque ao solo
        evts  = inputs.get("events", {})        # eventos do ciclo atual
        radio = inputs.get("radio", {})         # mensagem de rádio recebida

        # ── Simulação ─────────────────────────────────────────────────────────
        # valid: dados confiáveis neste ciclo (convenção geral de todos os grupos)
        sim_valid    = sim.get("valid", 0.0)
        phase        = sim.get("phase", 0.0)        # fase da simulação
        sim_time     = sim.get("sim_time", 0.0)     # tempo de simulação [s]
        exec_time    = sim.get("exec_time", 0.0)    # tempo de execução [s]

        # ── Ownship ───────────────────────────────────────────────────────────
        own_valid    = own.get("valid", 0.0)
        lat          = own.get("latitude", 0.0)         # latitude [deg]
        lon          = own.get("longitude", 0.0)        # longitude [deg]
        alt          = own.get("altitude", 0.0)         # altitude [m]
        spd          = own.get("ground_speed", 0.0)     # velocidade sobre o solo [m/s]
        trk          = own.get("ground_track", 0.0)     # trajetória sobre o solo [deg]
        hdg          = own.get("heading", 0.0)          # proa [deg]
        pit          = own.get("pitch", 0.0)            # arfagem [deg]
        rol          = own.get("roll", 0.0)             # rolamento [deg]
        terrain_elev = own.get("terrain_elev", 0.0)    # elevação do terreno sob a aeronave [m]
        dmg          = own.get("damage", 0.0)           # dano acumulado [0..1]
        pos_ecef     = own.get("pos_ecef", [])          # posição ECEF [m]
        vel_ecef     = own.get("vel_ecef", [])          # velocidade ECEF [m/s]
        vel_body     = own.get("vel_body", [])          # velocidade body [m/s]
        accel_ecef   = own.get("accel_ecef", [])        # aceleração ECEF [m/s²]
        accel_body   = own.get("accel_body", [])        # aceleração body [m/s²]
        euler_ned    = own.get("euler_ned", [])         # ângulos Euler NED [rad]
        euler_ecef   = own.get("euler_ecef", [])        # ângulos Euler ECEF [rad]
        ang_vel_body = own.get("ang_vel_body", [])      # vel. angular body [rad/s]
        ang_vel_ecef = own.get("ang_vel_ecef", [])      # vel. angular ECEF [rad/s]

        # ── Área de jogo ──────────────────────────────────────────────────────
        area_valid = area.get("valid", 0.0)
        area_dist  = area.get("boundary_dist", 0.0)    # distância para o limite [m]
        area_brg   = area.get("brg_to_center", 0.0)    # rumo para o centro [deg]

        # ── Território ────────────────────────────────────────────────────────
        terr_valid = terr.get("valid", 0.0)
        terr_dist  = terr.get("boundary_dist", 0.0)    # distância para o limite [m]
        terr_brg   = terr.get("brg_to_center", 0.0)    # rumo para o centro [deg]

        # ── FAOR ──────────────────────────────────────────────────────────────
        faor_valid  = faor.get("valid", 0.0)
        faor_dist   = faor.get("boundary_dist", 0.0)   # distância para o limite [m]; positivo = dentro
        faor_brg    = faor.get("brg_to_center", 0.0)   # rumo para o centro [deg]
        faor_active = faor.get("active", 0.0)           # 1.0 = FAOR ativa

        # ── Inventário ────────────────────────────────────────────────────────
        inv_valid = inv.get("valid", 0.0)
        fuel      = inv.get("fuel_remaining", 0.0)      # combustível restante [lbs]
        fuel_safe = inv.get("fuel_safe_amount", 0.0)    # mínimo seguro [lbs]
        nmsl      = inv.get("num_missiles", 0.0)        # mísseis disponíveis
        nbmb      = inv.get("num_bombs", 0.0)           # bombas disponíveis

        # ── Shared ────────────────────────────────────────────────────────────
        shrd_valid        = shrd.get("valid", 0.0)
        gnd_atk_start     = shrd.get("gnd_atk_start_time", -1.0)  # início do ataque ao solo [s]
        abort_gnd_atk     = shrd.get("abort_gnd_atk", 0.0)         # 1.0 = abortar ataque

        hold = shrd.get("hold", {})
        hold_num_acft     = hold.get("num_acft", 0.0)         # aeronaves no hold
        hold_my_idx       = hold.get("my_idx", -1.0)          # meu índice; -1 = fora do hold
        hold_ref_exec     = hold.get("ref_exec_time", -1.0)   # tempo de referência [s]
        hold_duration     = hold.get("ref_time_length", -1.0) # duração do hold [s]
        hold_start        = hold.get("start_time", -1.0)      # início do hold [s]

        cap = shrd.get("cap", {})
        cap_active        = cap.get("active", 0.0)             # 1.0 = CAP ativa
        cap_engaged       = cap.get("engagement_marker", 0.0)  # 1.0 = engajamento autorizado
        cap_start         = cap.get("start_time", -1.0)        # início da CAP [s]
        cap_lat           = cap.get("latitude", 0.0)           # centro da CAP [deg]
        cap_lon           = cap.get("longitude", 0.0)          # centro da CAP [deg]
        cap_hdg           = cap.get("heading", 0.0)            # proa da órbita [deg]
        cap_commit        = cap.get("commit_dist", 0.0)        # distância de commit [m]
        cap_cycle_end     = cap.get("flow_cycle_end", 0.0)     # 1.0 = fim do ciclo

        form = shrd.get("formation", {})
        form_ldr_wpt      = form.get("leader_fly_to", -1.0)   # waypoint do líder
        form_sign         = form.get("form_sign", 1.0)         # lado da formação (+1/-1)
        form_azimuth      = form.get("rel_azimuth", 0.0)       # azimute relativo [deg]
        form_dist         = form.get("rel_dist", 0.0)          # distância relativa [m]
        form_alt          = form.get("rel_alt", 0.0)           # altitude relativa [m]
        form_joined       = form.get("is_joined", 0.0)         # 1.0 = formação completa
        form_ldr_id       = form.get("leader_id", -1.0)        # player_id do líder
        aircraft          = form.get("aircraft", [])            # lista de aeronaves na formação
        # cada aeronave: rank, player_id, latitude, longitude, altitude,
        #                heading, airspeed, is_engaged, is_defending, is_joined, wing_id

        tgt_assign        = shrd.get("tgt_assign", [])         # atribuições de alvo
        # cada item: own_id, tgt_id, timer, engaged
        tgt_enabled_id    = shrd.get("tgt_assign_enabled_id", -1.0)  # id habilitado a atirar
        tgt_shot          = shrd.get("tgt_shot", [])            # tiros em andamento
        # cada item: own_id, tgt_id, lock_timer

        # ── Navegação ─────────────────────────────────────────────────────────
        nav_valid    = nav.get("valid", 0.0)
        wpt_idx      = nav.get("fly_to_idx", -1.0)             # índice do waypoint ativo
        wpt_lat      = nav.get("fly_to_lat", 0.0)              # latitude do waypoint [deg]
        wpt_lon      = nav.get("fly_to_lon", 0.0)              # longitude do waypoint [deg]
        wpt_elev     = nav.get("fly_to_elev", 0.0)             # elevação do waypoint [m]
        cmd_alt      = nav.get("fly_to_cmd_alt", alt)           # altitude comandada [m]
        cmd_spd      = nav.get("fly_to_cmd_speed", 200.0)      # velocidade comandada [m/s]
        brg          = nav.get("fly_to_bearing", 0.0)           # rumo para o waypoint [deg]
        dist         = nav.get("fly_to_distance", 0.0)          # distância até o waypoint [m]
        ttg          = nav.get("fly_to_ttg", 0.0)               # time-to-go [s]
        auto_seq     = nav.get("auto_seq", 0.0)                 # 1.0 = sequência automática

        nav_hold     = nav.get("hold", {})
        has_hold     = nav_hold.get("has_hold", 0.0)            # 1.0 = hold ativo
        hold_leg_hdg = nav_hold.get("leg_heading", 0.0)         # proa da perna [deg]
        hold_leg_t   = nav_hold.get("leg_time", 0.0)            # tempo da perna [s]
        hold_dur     = nav_hold.get("duration", 0.0)             # duração total [s]
        hold_ccw     = nav_hold.get("counter_clockwise", 0.0)   # 1.0 = anti-horário

        nav_terr     = nav.get("terrain", {})
        tf_leg       = nav_terr.get("tf_leg", 0.0)              # 1.0 = terrain following
        hat          = nav_terr.get("height_above_terrain", -1.0) # altura sobre o terreno [m]

        nav_wind     = nav.get("winding", {})
        has_wind     = nav_wind.get("has_winding", 0.0)         # 1.0 = winding ativo
        wind_tout    = nav_wind.get("timeout", 0.0)              # timeout [s]
        wind_angle   = nav_wind.get("angle", 0.0)               # ângulo de winding [deg]
        wind_spdf    = nav_wind.get("spd_factor", 1.0)          # fator de velocidade

        # ── Terreno à frente ──────────────────────────────────────────────────
        tdata_valid  = tdata.get("valid", 0.0)
        tpts         = tdata.get("points", [])
        # tpts[i]: elevação i segundos à frente; cada item: {"elev": float, "valid": float}
        # ex.: tpts[5].get("elev") → elevação 5 s à frente [m]

        # ── Radar ─────────────────────────────────────────────────────────────
        radar_valid  = radar.get("valid", 0.0)
        radar_mode   = radar.get("mode", 0.0)                   # AsaRadar::RadarMode como double

        # ── Air tracks ────────────────────────────────────────────────────────
        # cada track: iff_code, type, player_id,
        #   latitude, longitude, altitude,
        #   rel_azimuth, true_azimuth, elevation,
        #   range, gnd_range, gnd_track, gnd_speed, age,
        #   wez_max_o2t_true, wez_nez_o2t_true,   ← WEZ de nós sobre o alvo
        #   wez_max_t2o_true, wez_nez_t2o_true,   ← WEZ do alvo sobre nós (atual)
        #   wez_max_t2o_pred, wez_nez_t2o_pred,   ← WEZ do alvo sobre nós (predito)
        #   wez_pred_dist_corr, wez_nez_o2t_pred,
        #   has_spike, emitter_class, emitter_mode,
        #   pos_ecef, vel_ecef
        foes  = [t for t in trks if t.get("iff_code") == "foe"]
        frnds = [t for t in trks if t.get("iff_code") == "friend"]
        if foes:
            nearest  = min(foes, key=lambda t: t.get("range", float("inf")))
            tgt_rng  = nearest.get("range", 0.0)               # distância [m]
            tgt_brg  = nearest.get("true_azimuth", 0.0)        # rumo verdadeiro [deg]
            tgt_nez  = nearest.get("wez_nez_t2o_true", 0.0)    # NEZ do alvo sobre nós [m]

        # ── Ataque ao solo ────────────────────────────────────────────────────
        gatk_valid   = gatk.get("valid", 0.0)
        gatk_alt     = gatk.get("assigned_alt", -1.0)          # altitude de ataque [m]
        gatk_miss    = gatk.get("miss_dist", 0.0)              # distância de miss [m]
        gatk_toss    = gatk.get("toss_range", -1.0)            # alcance de toss [m]
        gatk_ete     = gatk.get("tgt_ete", -1.0)               # tempo até o alvo [s]
        gatk_next    = gatk.get("next_stpt", -1.0)             # próximo steerpoint

        # ── Eventos ───────────────────────────────────────────────────────────
        end_scan     = evts.get("end_of_rdr_scan", 0.0)        # 1.0 = fim do scan radar
        seeker_act   = evts.get("seeker_active", 0.0)           # 1.0 = seeker ativo
        msl_det      = evts.get("missile_detonated", 0.0)       # 1.0 = míssil detonado
        timeout_tgt  = evts.get("timeout_tgt_id", 0.0)          # id do alvo em timeout
        mission_done = evts.get("mission_accomplished", 0.0)    # 1.0 = missão cumprida
        msl_warn_taz = evts.get("outer_msl_warn_taz", [])       # azimutes de alerta [deg]

        # ── Rádio ─────────────────────────────────────────────────────────────
        has_msg      = radio.get("has_msg", 0.0)                # 1.0 = mensagem recebida
        voice_len    = radio.get("voice_time_length", 0.0)      # duração [s]
        voice_text   = radio.get("voice_content", "")           # conteúdo transcrito
        sender       = radio.get("sender_name", "")             # nome do remetente

        
        try:
            
            ##########################################################
            # self.log(json.dumps(inputs, default=str, indent=4))
            self.log(f"Fase: {phase}, Tempo: {sim_time:.1f}s, Altitude: {alt:.1f}m, Velocidade: {spd:.1f}m/s")
            
            
            # TODO: implementar lógica de decisão
            
        
            
            ##########################################################
            
        except Exception as e:
            self.log(f"Erro no predict(): {e}")
        

        return {
            "airspeed": np.array([[0]], dtype=np.float32),
            # TODO: incluir tantas outras saídas quantas forem necessárias, seguindo o mesmo formato
        }
        
    # -------------------------------------------------------------------------
    # Schema de saídas — declare o que seu predict() vai retornar
    # -------------------------------------------------------------------------

    def get_schema(self) -> ModelSchema:
        """Declara as saídas do modelo.

        Use ``self.output_spec(nome, ...)`` para cada tensor que ``predict()``
        vai retornar, e ``self._base_schema(outputs=[...])`` para montar o
        schema com as entradas padrão do agente já incluídas.

        Exemplos de saídas possíveis
        ----------------------------
        Comando de voo clássicos::
            self.output_spec("airspeed",    description="Velocidade [m/s]")

        Vetor intermediário para processamento no ASA/MIXR::

            self.output_spec("features", shape=[1, 8], description="..."),

        Ação discreta codificada como escalar::

            self.output_spec("action_id", description="Índice da ação selecionada"),
        """
        return self._base_schema(
            outputs=[
                self.output_spec("airspeed", description="Velocidade comandada [m/s]"),
                # TODO: declarar tantas outras saídas quantas forem necessárias, seguindo o mesmo formato
            ],
            description="Modelo de decisão do piloto autônomo"
        )