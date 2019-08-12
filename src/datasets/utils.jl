iscall(ex) = false
iscall(ex::Expr) = ex.head == :call

haskw(ex::Expr) = length(ex.args) >= 2 && isa(ex.args[2], Expr) && ex.args[2].head == :parameters

function get_first_arg(ex::Expr)
  !iscall(ex) && return nothing
  if haskw(ex)
    if length(ex.args) >= 3
      return ex.args[3]
    else
      return nothing
    end
  else
    if length(ex.args) >= 2
      return ex.args[2]
    else
      return nothing
    end
  end
end

macro try_take!(chn, ex=nothing)
  quote
    let v
      try
        v = take!($(esc(chn)))
      catch e
        if (isa(e, InvalidStateException) && e.state==:closed) ||
          (isa(e, RemoteException) && e.pid==$(esc(chn)).where && isa(e.captured.ex, InvalidStateException) && e.captured.ex.state==:closed)
          $(esc(ex))
        else
          rethrow()
        end
      end
    end
  end
end

macro try_call!(call::Expr, ex=nothing)
  chn = get_first_arg(call)
  quote
    let v
      try
        v = $(esc(call))
      catch e
        if (isa(e, InvalidStateException) && e.state==:closed) ||
          (isa(e, RemoteException) && e.pid==$(esc(chn)).where && isa(e.captured.ex, InvalidStateException) && e.captured.ex.state==:closed)
          $(esc(ex))
        else
          rethrow()
        end
      end
    end
  end
end

