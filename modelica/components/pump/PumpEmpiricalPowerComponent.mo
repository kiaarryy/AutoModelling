model PumpEmpiricalPowerComponent
  parameter Real P_nominal(unit="W") = 40000.0 annotation(Evaluate=false);
  parameter Real m_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real y_min(unit="1") = 0.05 annotation(Evaluate=false);
  parameter Real y_max(unit="1") = 1.20 annotation(Evaluate=false);
  parameter Real c0(unit="1") = 0.0 annotation(Evaluate=false);
  parameter Real c1(unit="1") = 0.0 annotation(Evaluate=false);
  parameter Real c2(unit="1") = 0.0 annotation(Evaluate=false);
  parameter Real c3(unit="1") = 1.0 annotation(Evaluate=false);
  parameter Real c4(unit="1") = 0.0 annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput m_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput y(unit="1");

  Modelica.Blocks.Interfaces.RealOutput P_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput m_flow_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealOutput y_s(unit="1");
  Modelica.Blocks.Interfaces.RealOutput phi(unit="1");

equation
  y_s = min(max(y, y_min), y_max);
  m_flow_s = max(0.0, m_flow_kg_s);
  phi = max(0.0, m_flow_s / max(1e-6, m_flow_nominal));
  P_W = max(0.0, P_nominal * (c0 + c1 * phi + c2 * y_s + c3 * y_s^3 + c4 * phi * y_s));

annotation(Documentation(info="<html><p>Signal-level empirical pump power component for generated system models. It mirrors the calibrated PumpEmpiricalPower FMU without external inputs from a table.</p></html>"));
end PumpEmpiricalPowerComponent;
