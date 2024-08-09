# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:04:12 2023

@author: HSJ
"""

class Custom_LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        hx = hx.squeeze(0)
        cx = cx.squeeze(0)

        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class Custom_LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        
        if len(input.shape) == 3:
            input = input.permute(1, 0, 2)
        
        inputs = input.unbind(0)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
            
        return torch.stack(outputs).permute(1, 0, 2), state


class Custom_MultiLayerLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = ModuleList(
            [Custom_LSTMLayer(Custom_LSTMCell, input_size, hidden_size) if i == 0
             else Custom_LSTMLayer(Custom_LSTMCell, hidden_size, hidden_size)
             for i in range(num_layers)]
        )

    @jit.script_method
    def forward(
        self, 
        input: Tensor, 
        state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        h, c = state
        new_h = []
        new_c = []
        
        for i, layer in enumerate(self.layers):
            input, (h_i, c_i) = layer(input, (h[i], c[i]))
            new_h.append(h_i)
            new_c.append(c_i)
        
        new_h = torch.stack(new_h, dim=0)
        new_c = torch.stack(new_c, dim=0)


        return input, (new_h, new_c)