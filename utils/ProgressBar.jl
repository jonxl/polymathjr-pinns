module ProgressBar
using ProgressMeter

# ---------------- progress bar ----------------

struct ProgressBarSettings
  maxiters::Int
  message::String
end

function Bar(s::ProgressBarSettings; step_size::Int=100, showvalues_fn::Function=() -> [])
  p_bar = Progress(s.maxiters, desc=s.message)
  position = Ref(0)
  last_displayed = Ref(-1)
  callback = function (_state, l)
    position[] = min(position[] + step_size, s.maxiters)
    if position[] != last_displayed[]
      ProgressMeter.update!(p_bar, position[]; showvalues=showvalues_fn())
      last_displayed[] = position[]
    end
    return false
  end
  return callback
end

export Bar

end
