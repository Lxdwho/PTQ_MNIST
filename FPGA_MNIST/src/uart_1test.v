module uart_try_main(
    input sys_clk,
    input sys_rst_n,
    input rx_data,
    output tx_data,
    output [7:0] led
    );

wire rx_done;
wire [7:0] send_data;
wire tx_busy;
wire [3:0] cnt;

uart_rx my_uart_rx(
    .sys_clk    (sys_clk),
    .sys_rst_n  (sys_rst_n),
    .rx_data    (rx_data),
    .rx_done    (rx_done),
    .send_data  (send_data)
);

assign led = send_data;

uart_tx my_uart_tx(
    .sys_clk    (sys_clk),
    .sys_rst_n     (sys_rst_n),
    .data        (send_data),
    .tx_en        (rx_done),
    .tx_data    (tx_data),
    .tx_busy    (tx_busy),
    .cnt        (cnt)
);

//ila_0 your_instance_name (
//    .clk(sys_clk), // input wire clk

//    .probe0(sys_rst_n), // input wire [0:0]  probe0  
//    .probe1(rx_data), // input wire [0:0]  probe1 
//    .probe2(rx_done), // input wire [0:0]  probe2 
//    .probe3(tx_data), // input wire [0:0]  probe3 
//    .probe4(tx_busy), // input wire [0:0]  probe4
//    .probe5(cnt) // input wire [3:0]  probe5
//);

endmodule