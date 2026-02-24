module ProgressBar
using ProgressMeter

# ---------------- progress bar ----------------

struct ProgressBarSettings
  maxiters::Int
  message::String
end

function Bar(s::ProgressBarSettings; step_size::Int=100)
  p_bar = Progress(s.maxiters, desc=s.message)
  position = Ref(0)
  callback = function (_state, l)
    position[] = min(position[] + step_size, s.maxiters)
    ProgressMeter.update!(p_bar, position[]; showvalues=[(:iter, position[]), (:loss, l)])
    return false
  end
  return callback
end

export Bar

end
