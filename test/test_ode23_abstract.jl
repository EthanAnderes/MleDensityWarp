include("../src/ode_tool.jl")
using ODE

tspan=[0,2*pi]
y0=Float64[1 0; 0 0]
function F(t, yy)
    y=yy[1]
    d0=[0 1; -1 0]*y[:,1]
    d1=[1,1]
    {[d0 d1],1}
end
#(T,Y)=ode23_abstract(F, tspan, {y0,0}) ## get samples from the whole ode path
(t1,y1)=ode23_abstract_end(F, tspan, {y0,0}) ## just get the ending value at the terminal time
