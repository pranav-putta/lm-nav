-- run configurations
overseer = require("overseer")
vim.keymap.set("n", "<leader>R1", function()
	-- Get current filepath
	local filepath = vim.fn.expand("%:p")

	if not filepath:match("%.yaml$") then
		print("Filepath doesn't end with .yaml!")
		return
	end

	-- Extract relevant session name and command details
	local pattern = "experiment/(.*)%.yaml"
	local window_name = filepath:match(pattern)

	if not window_name then
		print("Pattern after 'experiment/' not found in the filepath or it doesn't match the expected format!")
		return
	end

	-- Craft the shell commands
	local run_cmd = string.format(
		"zsh -c 'cd /srv/flash1/pputta7/ && mamba activate lmnav && python lmnav/bc_train.py %s'",
		window_name
	)
	-- Check if a window with the desired name exists
	local list_windows_command = "tmux list-windows -F '#W' | grep '^" .. window_name .. "$'"
	local existing_window = vim.fn.systemlist(list_windows_command)

	-- Craft tmux commands
	local create_window_and_run_command = "tmux new-window -n " .. window_name .. "`" .. run_cmd .. "`"
	local select_window_command = "tmux select-window -t " .. window_name
	local run_command_in_existing_window = "tmux send-keys -t " .. window_name .. " '" .. run_cmd .. "' Enter"

	-- Decide on actions based on the existence of the window
	if #existing_window == 0 then
		-- Window doesn't exist, create it
		vim.fn.system(create_window_and_run_command)
	else
		-- Window exists, send the command and attach to it
		vim.fn.system(run_command_in_existing_window)
		vim.fn.system(select_window_command)
	end
	-- Check if session exists, create and run if it doesn't, attach if it does
end, { desc = "Run BC" })
