`timescale 1ns / 1ps
module sim_uart_loop();

reg sys_clk;
reg sys_rst_n;
reg rx_data;
wire tx_data;


initial begin
    sys_clk <= 1'b0;
    sys_rst_n <= 1'b0;
    rx_data <= 1;
    #2000
    sys_rst_n <= 1'b1;
    #200
    rx_data <= 0;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 1;
    
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    #2170
    rx_data <= 0;
    #2170
    rx_data <= 1;
    
end

always # 5 sys_clk = ~sys_clk; 

uart_try_main my_uart_try_main(
    .sys_clk      (sys_clk  ),
    .sys_rst_n    (sys_rst_n),
    .rx_data      (rx_data  ),
    .tx_data      (tx_data  )
);

endmodule