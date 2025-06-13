module uart_tx(
    input sys_clk,
    input sys_rst_n,
    input [7:0] data,
    input tx_en,

    output reg tx_data,
    output tx_busy,
    output [3:0] cnt
    );

reg [7:0] baud_cnt;
reg [7:0] send_data;
reg [3:0] bit_cnt;
reg tx_flag;

assign tx_busy = tx_flag;
assign cnt = bit_cnt;
// 
always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        tx_flag <= 1'b0;
        send_data <= 8'd0;
        end
    else if(tx_en) begin
        tx_flag <= 1'b1;
        send_data <= data;
        end
    else if(tx_flag && bit_cnt == 4'd9 && baud_cnt == 8'd178)
        tx_flag <= 1'b0;
end

always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n)
        baud_cnt <= 8'd0;
    else if(tx_flag) begin
        if(baud_cnt < 8'd217 && tx_en && bit_cnt == 4'd9)
            baud_cnt <=  8'd0;
        else if(baud_cnt < 8'd217)
            baud_cnt <=  baud_cnt + 8'd1;
        else
            baud_cnt <=  8'd0;
    end
    else if(!tx_flag)
        baud_cnt <=  8'd0;
end

always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n)
        bit_cnt <= 4'd0;
    else if(tx_flag && baud_cnt == 8'd217 && bit_cnt < 9)
        bit_cnt <= bit_cnt + 4'd1;
    else if(tx_flag && bit_cnt == 4'd9 && baud_cnt == 8'd178)
        bit_cnt <= 4'd0;
    else if(tx_flag && bit_cnt == 4'd9 && baud_cnt >= 8'd178 && tx_en)
        bit_cnt <= 4'd0;
end

always @ (posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n)
        tx_data <= 1'b1;
    else if(tx_flag && bit_cnt < 4'd9 && baud_cnt == 8'd217) begin
        case(bit_cnt) // baud计数�?433时才执行，故�?后一位数未置�?
            0 : tx_data <= 1'b0;
            1 : tx_data <= send_data[0];
            2 : tx_data <= send_data[1];
            3 : tx_data <= send_data[2];
            4 : tx_data <= send_data[3];
            5 : tx_data <= send_data[4];
            6 : tx_data <= send_data[5];
            7 : tx_data <= send_data[6];
            8 : tx_data <= send_data[7];
            default : ;
        endcase
    end
    else if(tx_flag && bit_cnt === 4'd9 && baud_cnt == 8'd178)
        tx_data <= 1'b1;
end

endmodule